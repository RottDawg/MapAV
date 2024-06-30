"""
MapAV - Media Analysis Project

Main dependencies:
- opencv-python
- numpy
- moviepy
- matplotlib
- pillow
- psycopg2
- psutil
- tqdm
- torch
- deepspeech
- yolov5

To install these dependencies, you can use pip:
pip install opencv-python numpy moviepy matplotlib pillow psycopg2 psutil tqdm torch deepspeech
For YOLOv5: pip install yolov5

Note: Exact versions and additional setup may be required for some packages.
"""

import torch
import soundfile as sf
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import psycopg2
from PIL import Image
import io
import psutil
import logging
from tqdm import tqdm
import math
import json
import yolov5
import deepspeech
import tempfile

# Configure FFMPEG
ffmpeg_path = r"C:\Users\Andy\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe"
if os.path.exists(ffmpeg_path):
    change_settings({"FFMPEG_BINARY": ffmpeg_path})
else:
    print(f"FFMPEG not found at {ffmpeg_path}. Please check the path.")

# Load YOLOv5 model once
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Setup logging
logging.basicConfig(filename='media_analysis.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_SCORE = 1000  # Set a reasonable maximum score

# Rest of your code follows...
# Database setup
def setup_database():
    conn = psycopg2.connect(
        dbname="clipdb",
        user="admin",
        password="Ford1583EdgeST",
        host="192.168.1.199",
        port="5433"
    )
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS media_scores
                 (filename TEXT, file_type TEXT, duration REAL, 
                  avg_total_score REAL, avg_audio_score REAL, 
                  avg_visual_score REAL, avg_aesthetic_score REAL,
                  movement_scores JSONB, subject_tags JSONB,
                  transcript TEXT, word_count INTEGER)''')
    conn.commit()
    return conn

def is_anomalous(score):
    return math.isnan(score) or math.isinf(score) or score > MAX_SCORE

def check_and_log_anomalies(file_name, scores, score_type):
    anomalies = [i for i, score in enumerate(scores) if is_anomalous(score)]
    if anomalies:
        logging.warning(f"Anomalous {score_type} scores detected in {file_name} at indices: {anomalies}")
        valid_scores = [score for score in scores if not is_anomalous(score)]
        if valid_scores:
            mean_score = sum(valid_scores) / len(valid_scores)
        else:
            mean_score = 0
        for i in anomalies:
            scores[i] = mean_score
    return scores

def assess_visual_aesthetics(image):
    try:
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = image
        
        color_variance = np.std(image)
        brightness = np.mean(gray)
        brightness_score = 1 - abs(brightness - 128) / 128
        edges = cv2.Canny(gray, 100, 200)
        edge_score = np.sum(edges) / (image.shape[0] * image.shape[1])
        aesthetic_score = (color_variance * 0.4 + brightness_score * 0.3 + edge_score * 0.3)
        return aesthetic_score
    except Exception as e:
        logging.error(f"Error in assess_visual_aesthetics: {str(e)}")
        return 0

def assess_segment_quality(segment):
    try:
        if isinstance(segment, VideoFileClip):  # Video segment
            audio = segment.audio
            audio_array = audio.to_soundarray()
            audio_score = np.mean(np.abs(audio_array))

            frames = list(segment.iter_frames())
            frame_diffs = [np.mean(np.abs(frames[i] - frames[i-1])) for i in range(1, len(frames))]
            visual_change_score = np.mean(frame_diffs)
            
            aesthetic_scores = [assess_visual_aesthetics(frame) for frame in frames]
            aesthetic_score = np.mean(aesthetic_scores)
        else:  # Photo
            audio_score = 0  # Photos don't have audio
            visual_change_score = 0  # Photos don't have visual changes
            aesthetic_score = assess_visual_aesthetics(segment)

        return audio_score, visual_change_score, aesthetic_score
    except Exception as e:
        logging.error(f"Error in assess_segment_quality: {str(e)}")
        return 0, 0, 0

def detect_movement(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    movement_score = sum([cv2.contourArea(contour) for contour in contours])
    return movement_score

def analyze_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    movement_scores = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        movement_score = detect_movement(frame1, frame2)
        movement_scores.append(movement_score)
        frame1 = frame2

    cap.release()
    return movement_scores

def identify_subjects(file_path, model):
    try:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.ts')):
            # For video files, extract a frame
            video = cv2.VideoCapture(file_path)
            ret, frame = video.read()
            if not ret:
                raise ValueError("Could not read video file")
            # Save frame as temporary image
            temp_image_path = os.path.join(tempfile.gettempdir(), "temp_frame.jpg")
            cv2.imwrite(temp_image_path, frame)
            file_path = temp_image_path
        
        results = model(file_path, conf=0.25)  # Lowered confidence threshold
        subjects = results.pandas().xyxy[0]['name'].tolist()
        return list(set(subjects))  # Return unique subjects
    except Exception as e:
        print(f"Error in identify_subjects: {str(e)}")
        logging.error(f"Error in identify_subjects: {str(e)}")
        return []

def analyze_subjects(file_path, model):
    try:
        results = model(file_path)
        subjects = results.pandas().xyxy[0]['name'].tolist()
        return subjects
    except Exception as e:
        print(f"Error in analyze_subjects: {str(e)}")
        logging.error(f"Error in analyze_subjects: {str(e)}")
        return []

def transcribe_audio(audio_path):
    model = deepspeech.Model('C:\\DeepSpeech\\deepspeech-0.9.3-models.pbmm')
    audio = np.frombuffer(open(audio_path, 'rb').read(), np.int16)
    return model.stt(audio)

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mp4', '.wav')
    audio.write_audiofile(audio_path)
    return audio_path

def analyze_audio(audio_path):
    try:
        print(f"Analyzing audio: {audio_path}")
        model = deepspeech.Model('C:\\DeepSpeech\\deepspeech-0.9.3-models.pbmm')
        
        # Load audio file
        audio, sample_rate = sf.read(audio_path, dtype='int16')
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Perform speech recognition
        transcript = model.stt(audio)
        word_count = len(transcript.split())
        
        print(f"Audio analysis complete. Word count: {word_count}")
        return transcript, word_count
    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        logging.error(f"Error in analyze_audio: {str(e)}")
        return "", 0

def analyze_media(file_path, yolo_model, temp_dir, temp_files, segment_duration=2, audio_weight=0.3, visual_weight=0.3, aesthetic_weight=0.4):
    try:
        logging.info(f"Starting analysis of {file_path}")
        file_type = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        logging.info(f"File type: {file_type}")
        
        subjects = []
        transcript = ""
        word_count = 0
        movement_scores = []
        duration = 0
        segment_scores = []
        audio_scores = []
        visual_scores = []
        aesthetic_scores = []

        if file_type in ['.mp4', '.avi', '.mov', '.ts']:  # Video file
            logging.info("Analyzing video file")
            try:
                movement_scores = analyze_movement(file_path)
                logging.info(f"Movement analysis complete. Scores: {movement_scores[:5]}...")
            except Exception as e:
                logging.error(f"Error in movement analysis: {str(e)}")

            try:
                subjects = identify_subjects(file_path, yolo_model)
                logging.info(f"Subject analysis complete. Subjects: {subjects}")
            except Exception as e:
                logging.error(f"Error in subject analysis: {str(e)}")

            try:
                video = VideoFileClip(file_path)
                duration = video.duration

                # Extract audio to temp directory
                audio = video.audio
                audio_filename = file_name.replace(file_type, '.wav')
                audio_path = os.path.join(temp_dir, audio_filename)
                audio.write_audiofile(audio_path, codec='pcm_s16le')
                temp_files.append(audio_path)
                
                transcript, word_count = analyze_audio(audio_path)
                logging.info(f"Audio analysis complete. Word count: {word_count}")

                logging.info(f"Analyzing video segments. Total duration: {duration}")
                for i in range(0, int(duration), segment_duration):
                    segment = video.subclip(i, min(i + segment_duration, duration))
                    audio_score, visual_score, aesthetic_score = assess_segment_quality(segment)
                    total_score = (audio_score * audio_weight + 
                                   visual_score * visual_weight + 
                                   aesthetic_score * aesthetic_weight)
                    
                    segment_scores.append(total_score)
                    audio_scores.append(audio_score)
                    visual_scores.append(visual_score)
                    aesthetic_scores.append(aesthetic_score)

                video.close()
                logging.info("Video segment analysis complete")
            except Exception as e:
                logging.error(f"Error in video/audio analysis: {str(e)}")

        elif file_type in ['.jpg', '.jpeg', '.png', '.bmp']:  # Photo file
            logging.info("Analyzing photo file")
            try:
                with Image.open(file_path) as img:
                    photo = np.array(img)
                subjects = identify_subjects(file_path, yolo_model)
                logging.info(f"Subject analysis complete. Subjects: {subjects}")
                audio_score, visual_score, aesthetic_score = assess_segment_quality(photo)
                total_score = aesthetic_score  # For photos, total score is just the aesthetic score
                duration = 0  # Photos don't have duration
                segment_scores = [total_score]
                audio_scores = [audio_score]
                visual_scores = [visual_score]
                aesthetic_scores = [aesthetic_score]
                logging.info("Photo analysis complete")
            except Exception as e:
                logging.error(f"Error in photo analysis: {str(e)}")

        logging.info("Checking for anomalies")
        segment_scores = check_and_log_anomalies(file_name, segment_scores, "segment")
        audio_scores = check_and_log_anomalies(file_name, audio_scores, "audio")
        visual_scores = check_and_log_anomalies(file_name, visual_scores, "visual")
        aesthetic_scores = check_and_log_anomalies(file_name, aesthetic_scores, "aesthetic")
        movement_scores = check_and_log_anomalies(file_name, movement_scores, "movement")

        logging.info("Analysis complete")
        return duration, segment_scores, audio_scores, visual_scores, aesthetic_scores, movement_scores, subjects, transcript, word_count
    except Exception as e:
        logging.error(f"Error analyzing {file_path}: {str(e)}")
        return duration, segment_scores, audio_scores, visual_scores, aesthetic_scores, movement_scores, subjects, transcript, word_count

def file_already_analyzed(file_name):
    conn = setup_database()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM media_scores WHERE filename = %s", (file_name,))
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def plot_and_save_scores(file_path, segment_scores, audio_scores, visual_scores, aesthetic_scores, movement_scores):
    try:
        file_name = os.path.basename(file_path)
        plt.figure(figsize=(15, 10))
        plt.plot(segment_scores, label='Total Score')
        plt.plot(audio_scores, label='Audio Score')
        plt.plot(visual_scores, label='Visual Change Score')
        plt.plot(aesthetic_scores, label='Aesthetic Score')
        plt.plot(movement_scores, label='Movement Score')
        plt.xlabel('Segment Number')
        plt.ylabel('Score')
        plt.title(f'Media Analysis Scores - {file_name}')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        chart_path = os.path.splitext(file_path)[0] + '_analysis.png'
        plt.savefig(chart_path)
        plt.close()  # Close the plot to free up memory
        logging.info(f"Analysis chart saved: {chart_path}")
    except Exception as e:
        logging.error(f"Error saving plot for {file_path}: {str(e)}")

import os
from tkinter import filedialog, messagebox

def select_media_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Media Files",
        filetypes=[
            ("Media files", "*.mp4 *.avi *.mov *.ts *.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ],
        initialdir="C:/home/\\DS923plus/Z/test media"  # Set this to your default directory
    )
    
    if not file_paths:
        print("No files selected.")
        return []
    
    media_files = [path for path in file_paths if path.lower().endswith(('.mp4', '.avi', '.mov', '.ts', '.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not media_files:
        messagebox.showinfo("No Compatible Files", "No compatible media files were selected.")
        print("No compatible media files were selected.")
    else:
        print(f"Selected {len(media_files)} compatible files.")
    
    return media_files

def save_to_database(conn, filename, file_type, duration, avg_total_score, avg_audio_score, avg_visual_score, avg_aesthetic_score, movement_scores, subjects, transcript, word_count):
    try:
        c = conn.cursor()
        
        # Check if entry already exists
        c.execute("SELECT COUNT(*) FROM media_scores WHERE filename = %s", (filename,))
        if c.fetchone()[0] > 0:
            print(f"Entry for {filename} already exists. Updating...")
            c.execute('''UPDATE media_scores
                         SET file_type = %s, duration = %s, avg_total_score = %s, avg_audio_score = %s,
                             avg_visual_score = %s, avg_aesthetic_score = %s, movement_scores = %s,
                             subject_tags = %s, transcript = %s, word_count = %s
                         WHERE filename = %s''',
                      (file_type, duration, avg_total_score, avg_audio_score, avg_visual_score, avg_aesthetic_score,
                       json.dumps(movement_scores), json.dumps(subjects), transcript, word_count, filename))
        else:
            c.execute('''INSERT INTO media_scores
                         (filename, file_type, duration, avg_total_score, avg_audio_score, avg_visual_score, avg_aesthetic_score, movement_scores, subject_tags, transcript, word_count)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                      (filename, file_type, duration, avg_total_score, avg_audio_score, avg_visual_score, avg_aesthetic_score,
                       json.dumps(movement_scores), json.dumps(subjects), transcript, word_count))
        conn.commit()
        print(f"Data saved to database for {filename}")
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        logging.error(f"Error saving to database: {str(e)}")
        conn.rollback()

def view_database_contents(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM media_scores")
        rows = c.fetchall()
        
        print("\nDatabase Contents:")
        print("Filename | File Type | Duration | Avg Total Score | Avg Audio Score | Avg Visual Score | Avg Aesthetic Score | Movement Scores | Subject Tags | Transcript | Word Count")
        print("-" * 150)
        for row in rows:
            try:
                print(f"{row[0]} | {row[1]} | {row[2]} | " +
                      f"{row[3] if not is_anomalous(row[3]) else 'ANOMALY'} | " +
                      f"{row[4] if not is_anomalous(row[4]) else 'ANOMALY'} | " +
                      f"{row[5] if not is_anomalous(row[5]) else 'ANOMALY'} | " +
                      f"{row[6] if not is_anomalous(row[6]) else 'ANOMALY'} | " +
                      f"{row[7]} | {row[8]} | {row[9][:50]}... | {row[10]}")
            except Exception as e:
                print(f"Error displaying row: {row}")
                logging.error(f"Error displaying database row: {str(e)}")
    except Exception as e:
        logging.error(f"Error viewing database contents: {str(e)}")

def check_database_contents(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM media_scores")
        count = c.fetchone()[0]
        print(f"Number of entries in the database: {count}")
    except Exception as e:
        logging.error(f"Error checking database contents: {str(e)}")

def check_memory():
    if psutil.virtual_memory().percent > 90:
        logging.warning("High memory usage detected. Consider closing other applications or processing fewer files at once.")
        return False
    return True

def safe_mean(scores):
    valid_scores = [score for score in scores if not is_anomalous(score)]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

def test_database_connection(conn):
    try:
        c = conn.cursor()
        c.execute("INSERT INTO media_scores (filename, file_type, duration, avg_total_score, avg_audio_score, avg_visual_score, avg_aesthetic_score, movement_scores, subject_tags, transcript, word_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                  ("test.mp4", ".mp4", 0, 0, 0, 0, 0, '[]', '[]', "", 0))
        conn.commit()
        print("Test database insertion successful.")
    except Exception as e:
        print(f"Error in test database insertion: {str(e)}")

def cleanup_temp_files(temp_files):
    for file in temp_files:
        try:
            os.remove(file)
            print(f"Cleaned up: {file}")
        except Exception as e:
            print(f"Oops, couldn't clean up {file}: {str(e)}")

def cleanup_database(conn):
    try:
        c = conn.cursor()
        c.execute("DELETE FROM media_scores WHERE filename = 'test.mp4'")
        conn.commit()
        print("Cleaned up old test entries from database")
    except Exception as e:
        print(f"Error cleaning up database: {str(e)}")
        logging.error(f"Error cleaning up database: {str(e)}")
        
# Main execution
if __name__ == "__main__":
    try:
        # Set up temporary directory and file list
        temp_files = []
        temp_dir = tempfile.mkdtemp()

        print("Starting media analysis...")
        conn = setup_database()
        cleanup_database(conn)
        test_database_connection(conn)
        
        # Load YOLOv5 model once
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        media_paths = select_media_files()
        
        if media_paths:
            print(f"Selected {len(media_paths)} files for analysis.")
            for media_path in tqdm(media_paths, desc="Processing files", unit="file"):
                try:
                    result = analyze_media(media_path, yolo_model, temp_dir, temp_files)
                    if result is not None:
                        # Process and save results (your existing code here)
                        pass
                except Exception as e:
                    print(f"Error processing {media_path}: {str(e)}")
                    logging.error(f"Error processing {media_path}: {str(e)}")
            
            print("Analysis complete. Viewing database contents:")
            view_database_contents(conn)
        else:
            print("No files selected for analysis.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        logging.error(f"An unexpected error occurred: {str(e)}")
    
    finally:
        if 'conn' in locals():
            conn.close()
        cleanup_temp_files(temp_files)
        os.rmdir(temp_dir)