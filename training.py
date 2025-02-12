import torch
import os
from yolov5 import train

# Set paths (ensure correct formatting for Windows)
data_yaml = r"D:\Dsamp_ANPR\dataset\data.yaml"  # Path to dataset config
weights = r"D:\Dsamp_ANPR\yolov5\yolov5s.pt"  # Pre-trained model file
output_dir = r"D:\Dsamp_ANPR\yolov5\runs\train"

# Training parameters
epochs = 50  # Adjust based on needs
batch_size = 16
img_size = 640

# Set device to 'cpu' since you want to train on CPU only
device = 'cpu'  # Force CPU usage here, no condition

if __name__ == "__main__":  # Prevents unintended execution loops
    try:
        # Start Training
        train.run(
            data=data_yaml,
            weights=weights,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,  # Fix: 'img_size' -> 'imgsz'
            device=device  # Make sure the correct device is passed
        )

        # Locate trained model
        exp_folders = sorted([f for f in os.listdir(output_dir) if f.startswith("exp")], reverse=True)
        if exp_folders:
            saved_model_path = os.path.join(output_dir, exp_folders[0], "weights", "best.pt")  # Best model saved
            if os.path.exists(saved_model_path):
                print(f"✅ Training complete. Model saved at: {saved_model_path}")
            else:
                print("⚠️ Training finished, but 'best.pt' not found. Check training logs.")
        else:
            print("❌ No training output folder found. Something went wrong.")
    
    except Exception as e:
        print(f"⚠️ Error during training: {str(e)}")
