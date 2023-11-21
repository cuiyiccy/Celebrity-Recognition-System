from moviepy.video.io.VideoFileClip import VideoFileClip
import math
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
