
import argparse
import time
import cv2
import joblib
import numpy as np
from util import extract_features
from rejection import svm_is_unknown_from_probs, knn_is_unknown_from_mean_dist

# ------------------ LABELS ------------------
LABELS = {
    0: 'Cardboard',
    1: 'Glass',
    2: 'Metal',
    3: 'Paper',
    4: 'Plastic',
    5: 'Trash',
    6: 'Unknown'
}

# ------------------ FULL OBJECT (HAND INCLUDED) ------------------
def get_center_object_roi(frame, box_size=200):
    """
    Crop a fixed-size box in the middle of the frame.
    """
    h, w = frame.shape[:2]

    # Coordinates of the centered box
    x = w // 2 - box_size // 2
    y = h // 2 - box_size // 2
    bw = bh = box_size

    obj_crop = frame[y:y+bh, x:x+bw]
    white_bg = 255 * np.ones_like(frame)
    obj_resized = cv2.resize(obj_crop, (w, h))
    white_bg[:] = obj_resized

    return white_bg, (x, y, bw, bh)

# ------------------ MAIN LOOP ------------------
def run_live(model_path, scaler_path, svm_thr=0.55, knn_thr=1.0, camera_idx=0, box_size=200):
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    model_name = type(model).__name__.lower()
    is_svm = 'svc' in model_name or 'svm' in model_name
    is_knn = 'kneighborsclassifier' in model_name or 'knn' in model_name

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    prev = time.time()
    fps = 0.0

    print("[INFO] Place your object inside the green box.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # ---- FIXED CENTER ROI ----
        white_frame, bbox = get_center_object_roi(frame, box_size=box_size)

        # ---- FEATURES ----
        img_rgb = cv2.cvtColor(white_frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        feat = extract_features(img=img_resized)
        feat_s = scaler.transform([feat])

        pred_label = "Unknown"
        conf = 0.0

        # ---- PREDICTION ----
        if is_svm:
            probs = model.predict_proba(feat_s)[0]
            pred = int(model.predict(feat_s)[0])
            is_unk, score = svm_is_unknown_from_probs(probs, svm_thr)
            pred_label = LABELS[6] if is_unk else LABELS[pred]
            conf = score if is_unk else float(probs[pred])

        elif is_knn:
            pred = int(model.predict(feat_s)[0])
            dists, _ = model.kneighbors(feat_s, n_neighbors=model.n_neighbors)
            mean_dist = float(dists.mean())
            is_unk, _ = knn_is_unknown_from_mean_dist(mean_dist, knn_thr)
            pred_label = LABELS[6] if is_unk else LABELS[pred]
            conf = max(0.0, 1.0 - mean_dist / (knn_thr + 1e-6))

        # ---- FPS ----
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        # ---- DISPLAY ----
        text = f"{pred_label} ({conf:.2f}) FPS:{fps:.1f}"
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw the fixed green box in the center
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(display_frame, (x, y),
                          (x + bw, y + bh), (0, 255, 0), 2)

        cv2.imshow("MSI — Place Object in Box", display_frame)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ ENTRY ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--scaler', required=True)
    parser.add_argument('--svm_thr', type=float, default=0.55)
    parser.add_argument('--knn_thr', type=float, default=1.0)
    parser.add_argument('--camera_idx', type=int, default=0)
    parser.add_argument('--box_size', type=int, default=300, help="Size of the fixed box")
    args = parser.parse_args()

    run_live(args.model, args.scaler,
             args.svm_thr, args.knn_thr,
             args.camera_idx, args.box_size)