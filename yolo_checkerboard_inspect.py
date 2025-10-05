#!/usr/bin/env python3
import os, argparse, time, cv2, numpy as np, yaml, requests
from ultralytics import YOLO

# ---------- utils ----------
def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_checkerboard_scale(frame_bgr, cfg):
    cols, rows = int(cfg['pattern_cols']), int(cfg['pattern_rows'])
    square_mm = float(cfg['square_mm'])
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inner = (cols - 1, rows - 1)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, inner, flags)
    if not ret:
        return None, None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    objp = np.mgrid[0:inner[0], 0:inner[1]].T.reshape(-1, 2) * square_mm
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    obj_pts = objp.astype(np.float32)
    H, _ = cv2.findHomography(img_pts, obj_pts, 0)
    if H is None:
        return None, None

    h, w = gray.shape[:2]

    def mm_per_px(y):
        x = w / 2.0
        p1 = np.array([x, y, 1.0], np.float32)
        p2 = np.array([x + 1.0, y, 1.0], np.float32)
        q1 = H @ p1; q1 = q1[:2] / q1[2]
        q2 = H @ p2; q2 = q2[:2] / q2[2]
        return float(np.linalg.norm(q2 - q1))

    return H, mm_per_px

def minarea_rect_hw(mask):
    # รับ mask (0/255) -> สูง/กว้างของกรอบเอียงที่เล็กสุด
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    (w, h) = rect[1]
    box = cv2.boxPoints(rect).astype(int)
    return max(h, w), min(h, w), box  # บังคับให้ h>=w เสมอ

def grabcut_mask(bgr, iters=3):
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    rect = (2, 2, w - 4, h - 4)  # สมมุติวัตถุอยู่กลางกรอบ
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # เก็บชิ้นใหญ่สุดกัน noise
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(out); cv2.drawContours(clean, [c], -1, 255, -1)
    return clean

def map_height_to_size(h_mm, cfg):
    for lo, hi, ml in cfg['size_thresholds_mm']:
        if lo <= h_mm <= hi:
            return ml
    return None

def within(v, lohi):
    return (v >= lohi[0]) and (v <= lohi[1])

# ---------- main ----------
# --- เพิ่มด้านบนของไฟล์ (แทนที่ main() เดิมช่วงตั้งค่าโมเดล / วนลูป) ---
def is_detect_model(model):
    # Ultralytics: YOLO(...).task == 'detect' | 'segment' | 'classify' | 'pose'
    try:
        return getattr(model, "task", None) in ("detect", "segment")
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    default_cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser.add_argument("--config", default=default_cfg, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model = YOLO(cfg['yolo_model'])

    # ✅ ยืนยันว่าเป็นโมเดล detect/segment เท่านั้น
    if not is_detect_model(model):
        print("[ERROR] The loaded model is not a DETECTION model (looks like CLASSIFICATION).")
        print("        กรุณาใช้โมเดล detect (เช่น yolov8n.pt หรือ best.pt ที่ train ด้วย 'yolo detect train ...').")
        return

    cap = cv2.VideoCapture(cfg['camera_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['cap_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['cap_height'])

    have_scale = False
    send_api = False
    lastH = []

    print("[INFO] A=re-lock scale | S=API toggle | Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        disp = frame.copy()

        if not have_scale:
            H, mmppx = find_checkerboard_scale(frame, cfg)
            if H is not None:
                have_scale = True
                lastH.clear()
                cv2.putText(disp, "Scale locked (checkerboard)", (10, 40), 0, 1, (0, 255, 0), 2)

        cv2.putText(
            disp,
            f"Scale:{'OK' if have_scale else 'WAIT'} | API:{'ON' if send_api else 'OFF'}",
            (10, disp.shape[0]-10), 0, 0.7, (255,255,255), 2
        )

        if have_scale:
            res = model(disp, imgsz=cfg['yolo_imgsz'], conf=cfg['yolo_conf'], verbose=False)[0]

            # ✳️ ป้องกันกรณี boxes เป็น None (เช่นบางเวอร์ชัน หรือเป็นโมเดลไม่ใช่ detect)
            if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
                cv2.imshow("YOLO + Checkerboard + Geometry", disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')): break
                elif k in (ord('a'), ord('A')): have_scale=False; lastH.clear()
                elif k in (ord('s'), ord('S')): send_api = not send_api
                continue

            det = None
            for b in res.boxes:
                cls_id = int(b.cls[0].item())

                # โหมดเลือกคลาส: ใช้ bottle_class_id หรือรับทุกคลาสเป็นขวด
                accept_any = bool(cfg.get("accept_any_class", False))
                if (accept_any) or (cls_id == cfg['bottle_class_id']):
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    det = (x1, y1, x2, y2, float(b.conf[0].item()), cls_id)
                    break

            if det:
                x1,y1,x2,y2,conf,cls_id = det
                # ----- ส่วนเดิมคำนวณ geometry / grabcut / คำนวณ mm / วาดผล / ส่ง API -----
                roi = frame[max(0, y1):y2, max(0, x1):x2]
                h0, w0 = roi.shape[:2]
                mask = grabcut_mask(roi, iters=3)
                coverage = mask.sum()/255.0/(h0*w0 + 1e-6)
                blur = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                Hpx,Wpx,box = minarea_rect_hw(mask)
                if Hpx is None: Hpx,Wpx=(y2-y1),(x2-x1)
                else:
                    box += np.array([x1,y1]); cv2.polylines(disp,[box],True,(0,200,255),2)

                h_mm = Hpx * mmppx(y2)
                w_mm = Wpx * mmppx((y1+y2)/2.0)
                lastH.append(h_mm)
                if len(lastH) > cfg['median_N']: lastH.pop(0)
                h_med = float(np.median(np.array(lastH))) if lastH else h_mm
                size = map_height_to_size(h_med, cfg)

                width_ok = True
                if size and 'expected_width_mm' in cfg and str(size) in cfg['expected_width_mm']:
                    lo,hi = cfg['expected_width_mm'][str(size)]
                    width_ok = (w_mm >= lo) and (w_mm <= hi)

                aspect = h_mm/max(1e-6, w_mm)
                aspect_ok = 1.5 <= aspect <= 6.0
                ok_flag = (coverage>0.15) and (blur>30.0) and aspect_ok and (True if size is None else width_ok)

                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0) if ok_flag else (0,0,255),2)
                info1 = f"cls={cls_id} h={h_med:.1f}mm w={w_mm:.1f}mm ar={aspect:.2f} conf={conf:.2f}"
                info2 = f"size={size if size else '??'} ml  cover={coverage:.2f} blur={blur:.0f}"
                cv2.putText(disp, info1, (x1, y1-22), 0, 0.6, (255,255,255), 2)
                cv2.putText(disp, info2, (x1, y1-5), 0, 0.6, (0,255,0) if ok_flag else (0,0,255), 2)

                if send_api and ok_flag and size:
                    payload = {
                        "is_bottle": True, "is_pet": None, "size_ml": int(size),
                        "height_mm": round(h_med,1), "width_mm": round(w_mm,1),
                        "confidence": round(conf,3),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    try: requests.post(cfg['api_url'], json=payload, timeout=1.0)
                    except Exception as e:
                        cv2.putText(disp,f"API ERR:{e}",(10,60),0,0.6,(0,0,255),2)

        cv2.imshow("YOLO + Checkerboard + Geometry", disp)
        k=cv2.waitKey(1)&0xFF
        if k in (27, ord('q'), ord('Q')): break
        elif k in (ord('a'), ord('A')): have_scale=False; lastH.clear()
        elif k in (ord('s'), ord('S')): send_api = not send_api

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
