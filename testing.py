import torch 
import os
from yolov5 import detect

# Paths
weights_path = r"D:\Dsamp_ANPR\yolov5\runs\train\exp37\weights\best.pt"  # Path to the trained model
test_images_dir = r"D:\Dsamp_ANPR\dataset\test\images"  # Path to the test images
output_dir = r"D:\Dsamp_ANPR\yolov5\runs\detect\exp_test"  # Output directory for detection results

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set device to 'cpu' since you want to run inference on CPU only
device = 'cpu'  # Force CPU usage here, no condition

if __name__ == "__main__":  # Prevents unintended execution loops
    try:
        # Run detection
        detect.run(
            weights=weights_path,
            source=test_images_dir,
            project=output_dir,
            name='',  # Leave empty to save directly in the project directory
            exist_ok=True,  # Overwrite existing files
            device=device,  # Use CPU
            save_txt=True,  # Save results as .txt files
            save_conf=True,  # Save confidence scores
            save_crop=False,  # Do not save cropped images
            nosave=False,  # Save images with detections
            conf_thres=0.25,  # Confidence threshold
            iou_thres=0.45,  # IoU threshold
            max_det=1000,  # Maximum number of detections per image
            view_img=False,  # Do not show results
            classes=None,  # All classes
            agnostic_nms=False,  # Class-agnostic NMS
            augment=False,  # Augmented inference
            visualize=False,  # Visualize features
            line_thickness=3,  # Bounding box thickness
            hide_labels=False,  # Show labels
            hide_conf=False,  # Show confidence scores
            half=False  # Use FP16 half-precision inference
        )

        print(f"✅ Detection complete. Results saved at: {output_dir}")
    
    except Exception as e:
        print(f"⚠️ Error during detection: {str(e)}")