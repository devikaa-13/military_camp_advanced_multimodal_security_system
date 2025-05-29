import threading
import subprocess
import time
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import cv2
from scipy.io import wavfile
from ultralytics import YOLO

video_path = "/Users/devika/Downloads/What kind of shot group is this_.mp4"
wav_output_path = "audio.wav"

# === AUDIO THREAD: YAMNet Logic ===
def run_yamnet():
    print("🔊 [YAMNet] Extracting audio...")
    cmd = f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 -sample_fmt s16 "{wav_output_path}" -y'
    subprocess.run(cmd, shell=True, check=True)

    print("🔊 [YAMNet] Running audio classification...")
    model_audio = hub.load('https://tfhub.dev/google/yamnet/1')

    def get_filtered_predictions(scores_np, class_names, target_classes=None):
        if target_classes is None:
            target_classes = ["Gunshot, gunfire", "Explosion", "Machine gun", "Artillery fire"]

        target_indices = [class_names.index(cls) for cls in target_classes if cls in class_names]
        if not target_indices:
            return "None", 0.0

        target_scores = np.max(scores_np[:, target_indices], axis=0)
        max_index = np.argmax(target_scores)
        max_score = target_scores[max_index]
        threshold = 0.25
        if max_score > threshold:
            return target_classes[max_index], max_score
        else:
            return "None", max_score

    def class_names_from_csv(class_map_csv_text):
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    class_map_path = model_audio.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    sample_rate, wav_data = wavfile.read(wav_output_path, 'rb')
    if len(wav_data.shape) > 1:
        waveform = np.mean(wav_data, axis=1)
    else:
        waveform = wav_data

    waveform = waveform / tf.int16.max
    scores, _, _ = model_audio(waveform)
    scores_np = scores.numpy()
    predicted_class, confidence = get_filtered_predictions(scores_np, class_names)

    if confidence >= 0.25:
        print(f"🔊 Detected sound: {predicted_class} ({confidence:.2f})")
    else:
        print("🔊 No relevant sound detected.")

# === VIDEO THREAD: YOLO Video Processing ===
def run_yolo_video():
    print("🎥 [YOLO] Running object detection on video...")
    model_yolo = YOLO('/Users/devika/Downloads/best.pt')
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    yolo_outputs = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process every 15th frame to reduce computation
        if frame_id % 15 == 0:
            results = model_yolo.predict(frame, save=False, iou=0.7, conf=0.25)

            # Calculate timestamp based on frame count and FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = frame_id / fps
            time_str = time.strftime('%H:%M:%S', time.gmtime(timestamp))

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    prob = round(box.conf[0].item(), 2)
                    class_name = result.names[class_id]
                    yolo_outputs.append([
                        frame_id, time_str, x1, y1, x2, y2, class_id, class_name, prob
                    ])

        frame_id += 1

    cap.release()

    # Save detections to CSV
    df_yolo = pd.DataFrame(yolo_outputs, columns=[
        "frame", "timestamp", "x1", "y1", "x2", "y2", "class_id", "class_name", "confidence"
    ])
    df_yolo.to_csv("output_yolo_video.csv", index=False)
    print("✅ YOLO detections saved to output_yolo_video.csv")

# === PARALLEL EXECUTION ===
if __name__ == "__main__":
    # Create threads
    yolo_thread = threading.Thread(target=run_yolo_video)
    yamnet_thread = threading.Thread(target=run_yamnet)

    # Start threads
    yolo_thread.start()
    yamnet_thread.start()

    # Wait for both to complete
    yolo_thread.join()
    yamnet_thread.join()

    print("✅ Both YOLO and YAMNet processing finished!")