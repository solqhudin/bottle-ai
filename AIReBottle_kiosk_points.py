#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiosk Points (DB-ready):
- หน้าต่างกรอกรหัสใหญ่ + โชว์ชื่อ/แต้มรวมตลอดเวลา
- Detect PET bottle -> วัด mm -> map size -> ให้แต้ม
- ส่ง /recycle-api ด้วย payload ที่มี ext_id (uuid) + session/device/user
- รับแต้มรวมกลับมา หรือ fallback ไปเรียกสรุป
- Offline queue ถ้าเน็ตล่ม
"""

import os, argparse, time, json, cv2, numpy as np, yaml, requests, uuid
from pathlib import Path
from ultralytics import YOLO

# ------------------ basic utils ------------------
def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def api_url(base, route):
    return f"{base.rstrip('/')}/index.php?r={str(route).lstrip('?r=')}"

# ------------------ calibration ------------------
def find_checkerboard_scale(frame_bgr, cfg):
    cols, rows = int(cfg['pattern_cols']), int(cfg['pattern_rows'])
    square_mm = float(cfg['square_mm'])
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inner = (cols - 1, rows - 1)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, inner, flags)
    if not ret: return None, None
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    objp    = np.mgrid[0:inner[0],0:inner[1]].T.reshape(-1,2)*square_mm
    img_pts = corners.reshape(-1,2).astype(np.float32)
    obj_pts = objp.astype(np.float32)
    H,_ = cv2.findHomography(img_pts, obj_pts, 0)
    if H is None: return None, None

    h,w = gray.shape[:2]
    def mm_per_px(y):
        x=w/2.0
        p1=np.array([x,y,1.0],np.float32); p2=np.array([x+1.0,y,1.0],np.float32)
        q1=H@p1; q1=q1[:2]/q1[2]; q2=H@p2; q2=q2[:2]/q2[2]
        return float(np.linalg.norm(q2-q1))
    return H, mm_per_px

# ------------------ segmentation helpers ------------------
def grabcut_mask(bgr, iters=3):
    h,w=bgr.shape[:2]
    if h<4 or w<4: return np.zeros((h,w),np.uint8)
    mask=np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    rect=(2,2,w-4,h-4)
    bgd,fgd=np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
    cv2.grabCut(bgr,mask,rect,bgd,fgd,iters,cv2.GC_INIT_WITH_RECT)
    out=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    cnts,_=cv2.findContours(out,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return out
    c=max(cnts,key=cv2.contourArea)
    clean=np.zeros_like(out); cv2.drawContours(clean,[c],-1,255,-1)
    return clean

def minarea_rect_hw(mask):
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None,None,None
    c=max(cnts,key=cv2.contourArea)
    rect=cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    (w,h)=rect[1]
    box=cv2.boxPoints(rect).astype(int)
    return max(h,w), min(h,w), box  # h>=w

# ------------------ size mapping ------------------
def map_height_to_size(h_mm, cfg):
    for lo,hi,ml in cfg['size_thresholds_mm']:
        if lo<=h_mm<=hi: return ml
    return None

# ------------------ offline queue ------------------
def load_queue(path):
    if os.path.isfile(path):
        try: return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception: return []
    return []

def save_queue(path, items):
    try: Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception: pass

def try_flush_queue(cfg, session_token, student_id):
    qpath = cfg.get("queue_path", "C:/bottle-ai/outputs/offline_queue.json")
    base  = cfg.get("api_base", "").rstrip("/")
    route = cfg.get("record_route", "device/record_bottle")
    items = load_queue(qpath)
    if not items or not base: return 0
    remain=[]; sent=0
    for it in items:
        it.setdefault("session_token", session_token)
        it.setdefault("student_id", student_id)
        try:
            r=requests.post(api_url(base, route), json=it, timeout=3.0)
            ok=r.ok and r.json().get("ok", True)
        except Exception:
            ok=False
        if ok: sent+=1
        else: remain.append(it)
    save_queue(qpath, remain)
    return sent

# ------------------ API helpers ------------------
def start_session(cfg, student_id):
    """ขอ session + profile จาก backend
    คาดหวัง response:
    { ok:true,
      session_token:"...", device_id:1,
      user:{ id:1, student_id:"...", name:"...", total_points: 8 }
    }
    """
    base  = cfg.get("api_base"); dkey = cfg.get("device_api_key")
    route = cfg.get("start_session_route", "device/start_session")
    ttl   = int(cfg.get("session_ttl", 30))
    if not base or not dkey: return None, None, None, "missing api_base/device_api_key"
    try:
        r=requests.post(
            api_url(base, route),
            json={"device_api_key": dkey, "student_id": str(student_id), "ttl_minutes": ttl},
            timeout=6.0
        )
        j=r.json()
        if not j.get("ok"): return None, None, None, j.get("message","start_session failed")
        token=j.get("session_token")
        user=j.get("user",{})
        device_id=j.get("device_id")
        name=user.get("name") or str(student_id)
        total=user.get("total_points")
        return token, device_id, {"id":user.get("id"),"name":name,"student_id":str(student_id),"total":total}, None
    except Exception as e:
        return None, None, None, str(e)

def record_points(cfg, payload):
    """ส่งผลการรีไซเคิล 1 รายการ
    คาดหวังให้ backend:
      - สร้างแถวใน recycle_events (ใช้ ext_id ให้ยูนีค)
      - อัปเดต users.total_points ผ่าน trigger
      - คืน total_points ล่าสุดกลับมา
    Response ที่ดีที่สุด:
      { ok:true, total_points: 10 }
    """
    base  = cfg.get("api_base")
    route = cfg.get("record_route", "device/record_bottle")
    if not base: return False, "no api_base", None
    try:
        r=requests.post(api_url(base, route), json=payload, timeout=4.0)
        j=r.json() if r.ok else {}
        ok=r.ok and j.get("ok", True)
        total=j.get("total_points", None)
        return ok, j.get("message",""), total
    except Exception as e:
        return False, str(e), None

def fetch_total_points(cfg, session_token, student_id):
    base  = cfg.get("api_base")
    route = cfg.get("points_summary_route", "device/points_summary")
    if not base: return None
    try:
        r=requests.get(api_url(base, route),
                       params={"session_token": session_token, "student_id": student_id}, timeout=4.0)
        j=r.json() if r.ok else {}
        return j.get("total_points", None), j.get("name", None)
    except Exception:
        return None, None

# ------------------ UI helpers ------------------
def prompt_student_id_big():
    """หน้าต่างกรอกรหัส ขยายใหญ่ + ฟอนต์ใหญ่"""
    try:
        import tkinter as tk
        from tkinter import font as tkfont
        ok={}
        def go(): ok['v']=e.get().strip(); root.destroy()
        root=tk.Tk(); root.title("Recycle Kiosk – ลงชื่อเข้าใช้งาน")
        root.geometry("520x260+200+120")  # กว้างxสูง+posx+posy
        root.attributes("-topmost", True)
        f_title=tkfont.Font(size=18, weight="bold")
        f_label=tkfont.Font(size=14)
        f_entry=tkfont.Font(size=16)
        f_btn=tkfont.Font(size=14, weight="bold")

        tk.Label(root,text="กรอกรหัสนักศึกษา",font=f_title).pack(pady=12)
        e=tk.Entry(root,width=22,justify="center",font=f_entry)
        e.pack(pady=8); e.focus_set()
        tk.Button(root,text="เริ่มใช้งาน",command=go,font=f_btn,width=16).pack(pady=16)
        root.mainloop()
        return ok.get('v', None)
    except Exception:
        try: return input("Student ID: ").strip() or None
        except Exception: return None

def draw_header(frame, profile):
    """แถบหัวโชว์ชื่อ + แต้มรวม ตลอดเวลา"""
    if not profile: return
    h,w=frame.shape[:2]
    bar_h=36
    cv2.rectangle(frame,(0,0),(w,bar_h),(0,0,0),-1)
    text=f"{profile.get('name','')} ({profile.get('student_id','')})  |  Total: {profile.get('total','-')} pts"
    cv2.putText(frame,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

def draw_flash(frame, text, color, sec_left):
    if sec_left<=0: return
    h,w=frame.shape[:2]
    overlay=frame.copy()
    cv2.rectangle(overlay,(0,int(h*0.35)),(w,int(h*0.65)),(0,0,0),-1)
    cv2.addWeighted(overlay,0.45,frame,0.55,0,frame)
    cv2.putText(frame,text,(int(w*0.08),int(h*0.53)),0,1.0,color,2,cv2.LINE_AA)

# ------------------ main ------------------
def main():
    parser=argparse.ArgumentParser()
    default_cfg=os.path.join(os.path.dirname(__file__),"config.yaml")
    parser.add_argument("--config",default=default_cfg)
    args=parser.parse_args()

    cfg=load_cfg(args.config)

    # Step 0: login
    student_id=prompt_student_id_big()
    session_token=device_id=None
    profile=None
    if student_id:
        session_token, device_id, profile, err = start_session(cfg, student_id)
        if err: print("[WARN] start_session:", err)
        if not profile:
            # fallback ถ้า backend ไม่คืน profile
            total,name = fetch_total_points(cfg, session_token, student_id)
            profile={"id":None,"student_id":student_id,"name":name or student_id,"total":total}
    else:
        print("[INFO] ไม่ได้กรอกรหัส -> โหมดคิวออฟไลน์")

    # camera
    cap=cv2.VideoCapture(int(cfg.get("camera_index",0)), cv2.CAP_DSHOW)
    if not cap.isOpened(): cap=cv2.VideoCapture(int(cfg.get("camera_index",0)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("cap_width",1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(cfg.get("cap_height",720)))

    # model
    model=YOLO(cfg["yolo_model"])
    yolo_imgsz=int(cfg.get("yolo_imgsz",640))
    yolo_conf =float(cfg.get("yolo_conf",0.35))
    bottle_cls=int(cfg.get("bottle_class_id",39))
    accept_any=bool(cfg.get("accept_any_class",False))
    median_N=int(cfg.get("median_N",5))
    min_area=int(cfg.get("min_area_px",4000))
    min_ar  =float(cfg.get("min_ar",1.3))
    points_by_size={str(k):int(v) for k,v in cfg.get("points_by_size",{}).items()}
    send_cooldown=float(cfg.get("send_cooldown",1.5))
    queue_path=cfg.get("queue_path","C:/bottle-ai/outputs/offline_queue.json")

    have_scale=False; lastH=[]; last_send_ts=0.0
    flash_until=0.0; flash_text=""; flash_color=(80,220,80)
    last_size_sent=None

    print("[INFO] A=re-lock scale | Q=quit")

    while True:
        ok,frame=cap.read()
        if not ok: break
        disp=frame.copy()

        # header name + total
        draw_header(disp, profile)

        # scale lock
        if not have_scale:
            _,mmppx = find_checkerboard_scale(frame, cfg)
            if mmppx is not None:
                have_scale=True; lastH.clear()
                cv2.putText(disp,"Scale locked (checkerboard)",(10,60),0,1,(0,255,0),2)

        # status line
        status=f"Scale:{'OK' if have_scale else 'WAIT'}"
        cv2.putText(disp,status,(10,disp.shape[0]-10),0,0.7,(255,255,255),2)

        if have_scale:
            res=model(disp, imgsz=yolo_imgsz, conf=yolo_conf, verbose=False)[0]
            if getattr(res,"boxes",None) is None or len(res.boxes)==0:
                # idle: flush queue sometimes
                now=time.time()
                if (now-last_send_ts)>5.0:
                    flushed=try_flush_queue(cfg, session_token, student_id)
                    if flushed>0:
                        flash_text=f"Delivered {flushed} queued"; flash_color=(0,165,255); flash_until=now+1.2
                draw_flash(disp,flash_text,flash_color,max(0.0,flash_until-time.time()))
                cv2.imshow("Recycle Kiosk",disp)
                k=cv2.waitKey(1)&0xFF
                if k in (27,ord('q'),ord('Q')): break
                elif k in (ord('a'),ord('A')): have_scale=False; lastH.clear()
                continue

            det=None
            for b in res.boxes:
                cls_id=int(b.cls[0].item())
                if not (accept_any or cls_id==bottle_cls): continue
                x1,y1,x2,y2=map(int, b.xyxy[0].tolist())
                w,h=x2-x1, y2-y1
                if w*h<min_area: continue
                if (h/max(1,w))<min_ar: continue
                conf=float(b.conf[0].item())
                det=(x1,y1,x2,y2,conf); break

            if det:
                x1,y1,x2,y2,conf=det
                roi=frame[max(0,y1):y2, max(0,x1):x2]
                h0,w0=roi.shape[:2]
                mask=grabcut_mask(roi, iters=3)
                coverage=mask.sum()/255.0/(h0*w0+1e-6)
                blur=cv2.Laplacian(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                Hpx,Wpx,box=minarea_rect_hw(mask)
                if Hpx is None: Hpx,Wpx=(y2-y1),(x2-x1)
                else:
                    box+=np.array([x1,y1]); cv2.polylines(disp,[box],True,(0,200,255),2)

                h_mm=Hpx*mmppx(y2)
                w_mm=Wpx*mmppx((y1+y2)/2.0)
                lastH.append(h_mm)
                if len(lastH)>median_N: lastH.pop(0)
                h_med=float(np.median(np.array(lastH))) if lastH else h_mm
                size=map_height_to_size(h_med, cfg)

                width_ok=True
                if size and 'expected_width_mm' in cfg and str(size) in cfg['expected_width_mm']:
                    lo,hi=cfg['expected_width_mm'][str(size)]
                    width_ok=(w_mm>=lo) and (w_mm<=hi)

                aspect=h_mm/max(1e-6,w_mm)
                aspect_ok=1.5<=aspect<=6.0
                ok_flag=(coverage>0.15) and (blur>30.0) and aspect_ok and (True if size is None else width_ok)

                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0) if ok_flag else (0,0,255),2)
                cv2.putText(disp,f"h={h_med:.1f}mm w={w_mm:.1f}mm ar={aspect:.2f} conf={conf:.2f}",(x1,y1-22),0,0.6,(255,255,255),2)
                cv2.putText(disp,f"size={size if size else '??'} ml  cover={coverage:.2f} blur={blur:.0f}",(x1,y1-5),0,0.6,(0,255,0) if ok_flag else (0,0,255),2)

                now=time.time()
                if ok_flag and size:
                    pts=int(points_by_size.get(str(int(size)),0))
                    if (now-last_send_ts)>=send_cooldown and (last_size_sent!=size or (now-last_send_ts)>=send_cooldown*2):
                        # ---- payload ที่เข้า DB ได้จริง ----
                        payload={
                            "session_token": session_token,        # สำหรับตรวจสิทธิ์ + ผูก user/device
                            "student_id": student_id,              # กันพลาดฝั่ง server
                            "device_id": device_id,                # ระบุเครื่อง (ช่วย debug)
                            "ext_id": str(uuid.uuid4()),           # ยูนีคต่อรายการ -> map กับ recycle_events.ext_id
                            "size_ml": int(size),
                            "points": pts,
                            "bottles": 1,
                            "confidence": round(conf,3),
                            "height_mm": round(h_med,1),
                            "width_mm": round(w_mm,1),
                            "ts_client": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        ok,msg,total = record_points(cfg, payload)
                        if ok:
                            last_send_ts=now; last_size_sent=size
                            if profile: 
                                # ถ้า backend คืน total_points มาก็อัปเดตเลย
                                if total is not None: profile["total"]=total
                            flash_text=f"+{pts} pts | total = {profile['total'] if profile and profile.get('total') is not None else '-'}"
                            flash_color=(80,220,80); flash_until=now+1.2
                        else:
                            # offline queue
                            items=load_queue(queue_path); items.append(payload); save_queue(queue_path, items)
                            last_send_ts=now; last_size_sent=size
                            flash_text=f"+{pts} pts (queued {len(items)})"; flash_color=(0,165,255); flash_until=now+1.2

                        # ถ้า backend ไม่คืน total ให้ลองดึงสรุป
                        if profile and (profile.get("total") is None):
                            total,name = fetch_total_points(cfg, session_token, student_id)
                            if total is not None: profile["total"]=total
                            if name: profile["name"]=name

        draw_flash(disp,flash_text,flash_color,max(0.0,flash_until-time.time()))
        cv2.imshow("Recycle Kiosk", disp)
        k=cv2.waitKey(1)&0xFF
        if k in (27,ord('q'),ord('Q')): break
        elif k in (ord('a'),ord('A')): have_scale=False; lastH.clear()

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
