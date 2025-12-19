import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import random

# AUGMENTATIONS 
def augment_image(img):
    """Apply mild safe augmentations for waste images."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

#  FEATURE EXTRACTION 
def extract_features(image_path=None, img=None, resize_dim=(256, 256), augment=False):
    if img is None:
        if image_path is None:
            raise ValueError("Must provide either image_path or img array")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, resize_dim)
    if augment:
        img = augment_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    # Brightness & saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats.append(np.mean(hsv[:, :, 2]))           # brightness
    feats.append(np.mean(hsv[:, :, 1]))           # saturation mean
    feats.append(np.var(hsv[:, :, 1]))            # saturation variance

    # Edge strength
    feats.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    feats.append(np.mean(np.sqrt(sx**2 + sy**2)))
    edges = cv2.Canny(gray, 100, 200)
    feats.append(np.mean(edges))
    feats.append(np.var(edges))

    # Texture (LBP)
    for r in [1, 2]:
        n_points = 8 * r
        lbp = local_binary_pattern(gray, n_points, r, method='uniform')
        feats.append(np.mean(lbp))
        feats.append(np.var(lbp))

    # Color stats
    for i in range(3):
        channel = img[:, :, i]
        feats.append(np.mean(channel))
        feats.append(np.var(channel))

    # Shape features (glass / trash support) 

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        # Area
        area = cv2.contourArea(largest)

        # Bounding box & aspect ratio
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0

        # Extent (filled area ratio)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        feats.append(area)
        feats.append(aspect_ratio)
        feats.append(extent)

        # Hu Moments (shape descriptors)
        moments = cv2.moments(largest)
        hu_moments = cv2.HuMoments(moments).flatten()
        feats.extend(hu_moments)

    else:
        feats.extend([0, 0, 0])     # area, aspect, extent
        feats.extend([0] * 7)       # Hu moments

    # Edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    feats.append(edge_density)

    return np.array(feats)
