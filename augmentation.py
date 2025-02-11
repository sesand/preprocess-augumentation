import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

def augment_images_and_labels(input_dir, label_dir, output_img_dir, output_label_dir, num_augments=5):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomCrop(height=200, width=200, p=0.5),
        A.Affine(shear=(-15, 15), p=0.5),
        A.ToGray(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        ToTensorV2()
    ], bbox_params=A.Bbo
    Params(format='yolo', label_fields=['class_labels']))
    
    for img_name in tqdm([f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]):
        img_path = os.path.join(input_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)
        
        for i in range(num_augments):
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image'].permute(1, 2, 0).numpy()
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_img_dir, aug_name), aug_img)
            
            if bboxes:
                aug_label_path = os.path.join(output_label_dir, f"{os.path.splitext(img_name)[0]}_aug{i}.txt")
                with open(aug_label_path, "w") as f:
                    for cls, bbox in zip(class_labels, augmented['bboxes']):
                        f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

if __name__ == "__main__":
    input_folder = r"D:\Dsamp_ANPR\anpr-3\train\images"
    label_folder = r"D:\Dsamp_ANPR\anpr-3\train\labels"
    output_img_folder = r"D:\Dsamp_ANPR\augmentation_images"
    output_label_folder = r"D:\Dsamp_ANPR\augmentation_labels"
    augment_images_and_labels(input_folder, label_folder, output_img_folder, output_label_folder)
