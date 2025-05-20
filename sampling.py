import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse
import glob

sharpness_list = []

def is_sharp(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_value = cv2.Laplacian(image, cv2.CV_64F).var()
    sharpness_list.append(sharpness_value)
    return sharpness_value > threshold 

def interval_sampling(data_path, output_path, total_sample_frame):
    cap = cv2.VideoCapture(data_path)
    assert cap.isOpened(), "Cannot open video file"

    sample_path = os.path.join(output_path, "samples")

    # Calculate sampling interval
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, total_sample_frame, dtype=int)

    # Sample video
    for i in tqdm(indices, desc="Sampling Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            tqdm.write(f"Failed to read frame {i}")
            break
        else:
            filename = os.path.join(sample_path, f"frame_{i:04d}.jpg")
            cv2.imwrite(filename, frame)
            tqdm.write(f"Saved: {filename}")

    cap.release()
    print(f"\nExtraction complete.")


def laplacian_sampling_from_images(data_path, output_path, blur_threshold, save_blur):
    all_files = glob.glob(os.path.join(data_path, '*'))
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    sample_path = os.path.join(output_path, "samples")
    blur_path = os.path.join(output_path, "blur_samples")

    # Calculate sampling interval
    total_frames = len(image_files)

    saved_count = 0

    # Sample video
    for frame_count, image_path in enumerate(tqdm(image_files)):
        image = cv2.imread(image_path)
        if is_sharp(image, blur_threshold):
            filename = os.path.join(sample_path, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, image)
            saved_count += 1
            tqdm.write(f"Saved: {filename}")
        else:
            if save_blur:
                filename = os.path.join(blur_path, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, image)
            tqdm.write(f"Skipped: {filename}")

    print(f"\nSharpness list: {sorted(sharpness_list)}")
    print(f"\nExtraction complete. Saved {saved_count} frames.")


def laplacian_sampling_from_video(data_path, output_path, frame_rate, blur_threshold, max_frames, save_blur):
    cap = cv2.VideoCapture(data_path)
    assert cap.isOpened(), "Cannot open video file"

    sample_path = os.path.join(output_path, "samples")
    blur_path = os.path.join(output_path, "blur_samples")

    # Calculate sampling interval
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, fps // frame_rate)

    frame_count = 0
    saved_count = 0

    # Sample video
    for frame_count in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and saved_count >= max_frames:
            tqdm.write("MAX_FRAME reached extraction stopped")
            break

        if frame_count % frame_interval == 0:
            if is_sharp(frame, blur_threshold):
                filename = os.path.join(sample_path, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                tqdm.write(f"Saved: {filename}")
            else:
                if save_blur:
                    filename = os.path.join(blur_path, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(filename, frame)
                tqdm.write(f"Skipped: Frame {frame_count}")

    cap.release()
    print(f"\nExtraction complete. Saved {saved_count} frames.")


def main():
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Sample images from collections of images or video")
    method_subparsers = parser.add_subparsers(dest="sampling_method", help="Sampling method", required=True)
    parser.add_argument("--data", "-i", help="Path to data", type=str, required=True)
    parser.add_argument("--output-dir", "-o", help="Output directory", type=str, default=".")

    # Interval Sampling with Video
    interval_parser = method_subparsers.add_parser("interval", help="Sampling in interval")
    interval_parser.add_argument("--total-frame-sampling", "-tf", help="total frame sampling", type=int, default=226)

    # Laplacian Sampling
    laplacian_parser = method_subparsers.add_parser("laplacian", help="Sampling only images that passed threshold")
    input_subparser = laplacian_parser.add_subparsers(dest="input_type", help="Input type", required=True)
    laplacian_parser.add_argument("--blur-threshold", "-t", help="laplacian blur threshold", type=int, default=20)
    laplacian_parser.add_argument("--save-blur", help="laplacian blur threshold", type=bool, default=False)

    # Laplacian with Images
    image_subparser = input_subparser.add_parser("image", help="Accept collection of image")

    # Laplacian with Video
    video_subparser = input_subparser.add_parser("video", help="Accept video")
    video_subparser.add_argument("--max-frame", help="Max sampled iamge", type=int)
    video_subparser.add_argument("--frame-rate", help="Sample rate", type=int, default=2)

    args = parser.parse_args()

    print(args)

    # Open files check path validation
    assert os.path.isdir(args.output_dir), "Cannot find output directory"
    if args.sampling_method == "laplacian" and args.input_type == "image":
        assert os.path.isdir(args.data), "Cannot find input directory"
    else:
        assert os.path.isfile(args.data), "Cannot find input file"

    # Create output directory
    method_output = os.path.join(args.output_dir, args.sampling_method)
    sample_path = os.path.join(method_output, "samples")
    blur_path = os.path.join(method_output, "blur_samples")

    if not os.path.isdir(method_output):
        os.makedirs(method_output)
        print(f"create directory at {method_output}")

    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
        print(f"create directory at {sample_path}")

    if args.sampling_method == "laplacian" and args.save_blur:
        if not os.path.isdir(blur_path):
            os.makedirs(blur_path)
            print(f"create directory at {blur_path}")

    # Sampling Images
    if args.sampling_method == "laplacian":
        if args.input_type == "video":
            laplacian_sampling_from_video(args.data, method_output, args.frame_rate, args.blur_threshold, args.max_frame, args.save_blur)
        elif args.input_type == "image":
            laplacian_sampling_from_images(args.data, method_output, args.blur_threshold, args.save_blur)
        else:
            raise ValueError("Invalid input type spacification")
    elif args.sampling_method == "interval":
            interval_sampling(args.data, method_output, args.total_frame_sampling)
    else:
        raise ValueError("Invalid Sampling Method")



if __name__ == "__main__":
    main()
