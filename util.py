from moviepy.video.io.VideoFileClip import VideoFileClip
from facenet_pytorch import MTCNN
from pathlib import Path
import math
import cv2
import os


def split_video(video_path, output_path, segment_length=5):
    video = VideoFileClip(video_path)
    video_duration = int(video.duration)
    num_segments = math.ceil(video_duration / segment_length)
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, video_duration)
        subclip = video.subclip(start_time, end_time)
        original_name, ext = os.path.splitext(os.path.basename(video_path))
        output_filename = f"{original_name}_segment_{i + 1}{ext}"
        output_filepath = os.path.join(output_path, output_filename)
        subclip.write_videofile(output_filepath, codec='libx264')
        print(f"Segment {i + 1} done: {output_filepath}")


def save_screenshots(video_path, output_folder, interval=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    screenshot_count = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            screenshot_filename = os.path.join(output_folder, f'screenshot_9_{screenshot_count}.jpg')
            cv2.imwrite(screenshot_filename, frame)
            print(f'Screenshot saved: {screenshot_filename}')
            screenshot_count += 1
        current_frame += int(fps * interval)
    cap.release()


def process_images(input_folder, output_folder, results_file, cropped_results_file):
    mtcnn = MTCNN(keep_all=True, image_size=224, thresholds=[0.4, 0.5, 0.5], min_face_size=60)
    Path(output_folder).mkdir(exist_ok=True)
    files = [f.name for f in Path(input_folder).glob('*.jpg')]
    with open(results_file, 'w') as f, open(cropped_results_file, 'w') as f1:
        for filename in files:
            img = cv2.imread(f'{input_folder}/{filename}')
            faces, _ = mtcnn.detect(img)
            if faces is not None:
                for i, box in enumerate(faces):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    cropped_face = img[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        face_filename = f'cropped_9_{filename[:-4]}_face{i}.jpg'
                        cv2.imwrite(f'{output_folder}/{face_filename}', cropped_face)
                        label = 1 if "mp" in filename else 0
                        f1.write(f'{face_filename},{label}\n')
            label = 1 if "mp" in filename else 0
            f.write(f'{filename},{label}\n')


def process_videos_in_folder(video_folder, screenshots_folder, cropped_folder, results_file, cropped_results_file):
    for video_file in Path(video_folder).glob('*.mp4'):
        print(f'Processing video: {video_file}')
        save_screenshots(str(video_file), screenshots_folder)
        process_images(screenshots_folder, cropped_folder, results_file, cropped_results_file)
