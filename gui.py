import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from collections import deque
import numpy as np
from moviepy.editor import VideoFileClip
from pytube import YouTube

# Assuming these are defined in your environment
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

# Replace with your actual model and classes list
# LRCN_model = ...
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "Fencing", "Biking", "PlayingPiano", "Skiing"]

def download_youtube_video(url, output_dir):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4', progressive=True).first()
    video.download(output_dir)
    return yt.title

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        video_path.set(file_path)
        messagebox.showinfo("Selected Video", f"Selected video: {os.path.basename(file_path)}")

def predict_activity():
    input_file_path = video_path.get()
    if not input_file_path:
        messagebox.showerror("Error", "Please select a video file first.")
        return

    output_file_path = os.path.join(output_dir, f"{os.path.basename(input_file_path).split('.')[0]}-Output-SeqLen{SEQUENCE_LENGTH}.mp4")
    predict_on_video(input_file_path, output_file_path, SEQUENCE_LENGTH)

    VideoFileClip(output_file_path, audio=False, target_resolution=(300, None)).ipython_display()
    messagebox.showinfo("Prediction Complete", f"Prediction complete. Output video saved to: {output_file_path}")

# Setup Tkinter GUI
root = tk.Tk()
root.title("Human Activity Recognition")

video_path = tk.StringVar()

tk.Label(root, text="Video File Path:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=video_path, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=browse_video).grid(row=0, column=2, padx=10, pady=10)
tk.Button(root, text="Predict Activity", command=predict_activity).grid(row=1, column=1, padx=10, pady=10)

output_dir = "test_videos"
os.makedirs(output_dir, exist_ok=True)

root.mainloop()
