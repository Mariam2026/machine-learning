
import os
import shutil
import argparse
import cv2
from pathlib import Path

def check_image(img_path):
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
     
        _ = img.shape
        return True
    except Exception:
        return False

def clean_dataset(input_dir, output_dir=None, delete_corrupted=False):
   
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Copying clean images to: {output_dir}")
    elif delete_corrupted:
        print(f"[WARNING] Will DELETE corrupted images from: {input_dir}")
        confirm = input("Are you sure? Type 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("[INFO] Cancelled.")
            return
    
    # Get all class folders
    class_dirs = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    
    total_images = 0
    corrupted_images = 0
    good_images = 0
    
    corrupted_list = []
    
    print(f"\n[INFO] Scanning dataset...")
    
    
    for cls in sorted(class_dirs):
        cls_input_path = os.path.join(input_dir, cls)
        
        if output_dir:
            cls_output_path = os.path.join(output_dir, cls)
            os.makedirs(cls_output_path, exist_ok=True)
        
        cls_good = 0
        cls_corrupted = 0
        
        # Check all images in this class
        for fname in os.listdir(cls_input_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img_path = os.path.join(cls_input_path, fname)
            total_images += 1
            
            if check_image(img_path):
                # Good image
                good_images += 1
                cls_good += 1
                
                if output_dir:
                    # Copy to clean dataset
                    shutil.copy2(img_path, os.path.join(cls_output_path, fname))
            else:
                # Corrupted image
                corrupted_images += 1
                cls_corrupted += 1
                corrupted_list.append(img_path)
                
                if delete_corrupted and not output_dir:
                    # Delete corrupted image
                    try:
                        os.remove(img_path)
                        print(f"[DELETED] {img_path}")
                    except Exception as e:
                        print(f"[ERROR] Could not delete {img_path}: {e}")
        
        print(f"Class '{cls}': {cls_good} good, {cls_corrupted} corrupted")
    
    
    print(f"\n[SUMMARY]")
    print(f"Total images scanned: {total_images}")
    print(f"Good images: {good_images} ({good_images*100/total_images:.1f}%)")
    print(f"Corrupted images: {corrupted_images} ({corrupted_images*100/total_images:.1f}%)")
    
    if output_dir:
        print(f"\nClean dataset saved to: {output_dir}")
        print(f"You can now train using: --dataset {output_dir}")
    elif delete_corrupted:
        print(f"\nCorrupted images deleted from: {input_dir}")
    else:
        print(f"\n[INFO] Found {corrupted_images} corrupted images.")
        print(f"[INFO] To remove them, run again with --delete flag")
    
    # Save list of corrupted files
    if corrupted_list:
        with open('corrupted_images.txt', 'w') as f:
            for path in corrupted_list:
                f.write(path + '\n')
        print(f"\n[INFO] List of corrupted images saved to: corrupted_images.txt")
    
    return good_images, corrupted_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean dataset by removing corrupted images'
    )
    parser.add_argument('--dataset', required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--output', default=None,
                       help='Output directory for clean dataset (if not provided, works in-place)')
    parser.add_argument('--delete', action='store_true',
                       help='Delete corrupted images from original dataset (use with caution!)')
    
    args = parser.parse_args()
    
  
    print("DATASET CLEANING TOOL")
    
    
    good, corrupted = clean_dataset(args.dataset, args.output, args.delete)
    
    if corrupted > 0:
        print(f" You have {corrupted} corrupted images! ")
        print(f" This may reduce accuracy by ~{corrupted*0.05:.1f}%")
        print(f" Recommendation: ")
        if not args.output and not args.delete:
            print(f" Run: python fix_dataset.py --dataset {args.dataset} --output dataset_clean ")
            print(f" Then train with: --dataset dataset_clean ")
    else:
        print(" All images are readable! Your dataset is clean. ")
