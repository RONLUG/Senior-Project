import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def gradient_edge_indicator(image_path, alpha=1.0, save_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    gei = 1 / (1 + grad_magnitude**2)

    # Threshold GEI to detect edges
    edge_mask = gei < 0.8

    output_image = np.where(edge_mask[..., None], np.clip(alpha * image, 0, 255), image).astype(np.uint8)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, output_image)

def process_folder_with_gei(input_folder, output_folder, alpha=1.0):
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(supported_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                gradient_edge_indicator(input_path, alpha=alpha, save_path=output_path)
                tqdm.write(f"Processed: {filename}")
            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")

def main():
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Sample images from collections of images or video")
    parser.add_argument("--data", "-i", help="Output directory", type=str, required=True)
    parser.add_argument("--output-dir", "-o", help="Output directory", type=str, default=".")

    args = parser.parse_args()

    print(args)

    # Open files check path validation
    assert os.path.isdir(args.data), "Cannot find input directory"
    assert os.path.isdir(args.output_dir), "Cannot find output directory"

    # Create output directory
    save_dir = os.path.join(args.output_dir, "edge_enhanced")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print(f"create directory at {save_dir}")

    process_folder_with_gei(args.data, save_dir, alpha=1.0)



if __name__ == "__main__":
    main()

