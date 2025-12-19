import os
import numpy as np
import argparse
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from util import extract_features
import cv2
import random
import tempfile

# AUGMENTATION FUNCTIONS
def augment_image(img):
    """Apply random augmentation: flip, rotate, scale, brightness."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    scale = random.uniform(0.9, 1.1)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def extract_features_from_array(img_array):
    tmp_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_path, img_array)
    feats = extract_features(tmp_path)
    os.remove(tmp_path)
    return feats

# DATA LOADING
def load_dataset(base_path):
    X, y = [], []
    class_map = {}
    classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)
        class_map[idx] = cls
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for i, img in enumerate(imgs, 1):
            img_path = os.path.join(cls_path, img)
            try:
                X.append(extract_features(img_path))
                y.append(idx)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")
            if i % 50 == 0:
                print(f"[INFO] Processed {i}/{len(imgs)} images in class '{cls}'")
    return np.array(X), np.array(y), class_map, classes

# TRAINING AUGMENTATION
def augment_training_set(X_train, y_train, base_path, classes, target_per_class):
    X_aug, y_aug = [], []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        n_current = sum(1 for label in y_train if label == idx)
        n_aug = 0
        while n_current < target_per_class:
            img_name = random.choice(imgs)
            img_path = os.path.join(cls_path, img_name)
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                continue
            img_aug = augment_image(img_cv)
            try:
                feats = extract_features_from_array(img_aug)
                X_aug.append(feats)
                y_aug.append(idx)
                n_current += 1
                n_aug += 1
            except:
                continue
        print(f"[INFO] Augmented {n_aug} images for class '{cls}'")
    X_train = np.vstack([X_train, np.array(X_aug)])
    y_train = np.hstack([y_train, np.array(y_aug)])
    return X_train, y_train

# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset folder')
    parser.add_argument('--out_dir', required=True, help='Path to save models')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--target_per_class', type=int, default=None, help='Target number of images per class after augmentation')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf'], help='SVM kernel')
    parser.add_argument('--C', type=float, default=12.0, help='SVM regularization parameter')
    parser.add_argument('--knn_neighbors', type=int, default=5, help='Number of neighbors for k-NN')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading dataset without augmentation...")
    X, y, class_map, classes = load_dataset(args.dataset)
    print(f"[INFO] Total samples: {len(X)} | Feature length: {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if args.augment and args.target_per_class:
        X_train, y_train = augment_training_set(X_train, y_train, args.dataset, classes, args.target_per_class)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # SVM
    svm = SVC(
        kernel=args.kernel,
        C=args.C,
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    print("[INFO] Training SVM...")
    svm.fit(X_train, y_train)

    # Print SVM Training Accuracy
    y_train_pred_svm = svm.predict(X_train)
    train_acc_svm = accuracy_score(y_train, y_train_pred_svm)
    print(f"[TRAINING RESULT] SVM Training Accuracy: {train_acc_svm:.4f}")

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=args.knn_neighbors)
    print("[INFO] Training k-NN...")
    knn.fit(X_train, y_train)

    # Print k-NN Training Accuracy
    y_train_pred_knn = knn.predict(X_train)
    train_acc_knn = accuracy_score(y_train, y_train_pred_knn)
    print(f"[TRAINING RESULT] k-NN Training Accuracy: {train_acc_knn:.4f}")

    # Validation
    y_pred_svm = svm.predict(X_val)
    acc_svm = accuracy_score(y_val, y_pred_svm)
    print(f"\n[RESULT] SVM Validation Accuracy: {acc_svm:.4f}\n")
    print(classification_report(y_val, y_pred_svm, target_names=class_map.values()))

    y_pred_knn = knn.predict(X_val)
    acc_knn = accuracy_score(y_val, y_pred_knn)
    print(f"\n[RESULT] k-NN Validation Accuracy: {acc_knn:.4f}\n")
    print(classification_report(y_val, y_pred_knn, target_names=class_map.values()))

    # Save models
    joblib.dump(svm, os.path.join(args.out_dir, "svm.joblib"))
    joblib.dump(knn, os.path.join(args.out_dir, "knn.joblib"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    print("[INFO] Models saved.")

if __name__ == "__main__":
    main()
