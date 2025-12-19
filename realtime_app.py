import argparse
import time
import cv2
import joblib
import numpy as np

from util import extract_features
from rejection import svm_is_unknown_from_probs, knn_is_unknown_from_mean_dist

LABELS = {
    0: 'Glass Bottle',
    1: 'Paper Sheet',
    2: 'Cardboard Box',
    3: 'Plastic Bottle',
    4: 'Metal Can',
    5: 'Trash Bag',
    6: 'Unknown'
}

class ObjectTracker:
    """Smooth bounding box tracking."""
    def __init__(self, alpha=0.3):
        self.prev_box = None
        self.alpha = alpha

    def smooth_box(self, box):
        if self.prev_box is None:
            self.prev_box = box
            return box
        tl_prev, br_prev = self.prev_box
        tl_new = (int(self.alpha*box[0][0] + (1-self.alpha)*tl_prev[0]),
                  int(self.alpha*box[0][1] + (1-self.alpha)*tl_prev[1]))
        br_new = (int(self.alpha*box[1][0] + (1-self.alpha)*br_prev[0]),
                  int(self.alpha*box[1][1] + (1-self.alpha)*br_prev[1]))
        self.prev_box = (tl_new, br_new)
        return (tl_new, br_new)

def run_live(model_path, scaler_path, svm_thr=0.55, knn_thr=1.0, camera_idx=0):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    model_name = type(model).__name__.lower()
    is_svm = 'svc' in model_name or 'svm' in model_name
    is_knn = 'kneighborsclassifier' in model_name or 'knn' in model_name

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    prev_time = time.time()
    fps = 0.0
    tracker = ObjectTracker(alpha=0.3)

    # Background subtractor for better object detection
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # ---------- PREDICTION ----------
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        feat = extract_features(img=img_resized)
        feat_s = scaler.transform([feat])

        pred_label = 'Unknown'
        conf = 0.0

        if is_svm:
            probs = model.predict_proba(feat_s)[0]
            pred = int(model.predict(feat_s)[0])
            is_unk, score = svm_is_unknown_from_probs(probs, svm_thr)
            pred_label = 'Unknown' if is_unk else LABELS[pred]
            conf = score if is_unk else float(probs[pred])
        elif is_knn:
            pred = int(model.predict(feat_s)[0])
            dists, _ = model.kneighbors(feat_s, n_neighbors=model.n_neighbors)
            mean_dist = float(dists.mean())
            is_unk, md = knn_is_unknown_from_mean_dist(mean_dist, knn_thr)
            pred_label = 'Unknown' if is_unk else LABELS[pred]
            conf = max(0.0, 1.0 - (md / (knn_thr + 1e-6))) if is_unk else max(0.0, 1.0 - (mean_dist / (knn_thr + 1e-6)))
        else:
            try:
                probs = model.predict_proba(feat_s)[0]
                pred = int(model.predict(feat_s)[0])
                is_unk, score = svm_is_unknown_from_probs(probs, svm_thr)
                pred_label = 'Unknown' if is_unk else LABELS[pred]
                conf = float(probs.max())
            except Exception:
                pred = int(model.predict(feat_s)[0])
                pred_label = LABELS[pred]
                conf = 1.0

        # ---------- OBJECT DETECTION ----------
        fg_mask = back_sub.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box = None
        if contours:
            # Take the largest moving object
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:  # ignore small noise
                x, y, w, h = cv2.boundingRect(largest)
                size = max(w, h)
                cx, cy = x + w//2, y + h//2
                half = size // 2
                top_left = (max(cx - half, 0), max(cy - half, 0))
                bottom_right = (min(cx + half, display_frame.shape[1]-1),
                                min(cy + half, display_frame.shape[0]-1))
                box = (top_left, bottom_right)

        # ---------- DRAW GREEN BOX ----------
        if box:
            box = tracker.smooth_box(box)
            cv2.rectangle(display_frame, box[0], box[1], (0, 255, 0), 2)

        # ---------- FPS ----------
        now = time.time()
        curr_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        fps = 0.9*fps + 0.1*curr_fps
        prev_time = now

        # ---------- DISPLAY LABEL ----------
        text = f"{pred_label} ({conf:.2f})  FPS:{fps:.1f}"
        cv2.putText(display_frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Object Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--scaler', required=True)
    parser.add_argument('--svm_thr', type=float, default=0.55)
    parser.add_argument('--knn_thr', type=float, default=1.0)
    parser.add_argument('--camera_idx', type=int, default=0)
    args = parser.parse_args()
    run_live(args.model, args.scaler, args.svm_thr, args.knn_thr, args.camera_idx)