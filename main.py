# pip install opencv-python
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Define constants
VIDEO_WIDTH = 420
VIDEO_HEIGHT = 300

# Initialize Mediapipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the trained CNN model
model = tf.keras.models.load_model('sign_language_cnn_model_word50.h5')

# Load the LabelEncoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes50.npy')

# Define image size (same as during model training)
IMG_SIZE = 64

# Global variables for prediction
current_word = None
start_time = None
sentence = ""
displayed_word = ""
last_sentence = ""
running = False
is_paused = False
current_word_idx = 0
words_list = []
video_paths = []
video_thread = None

# Function to preprocess the image frame
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

# Function to predict the sign from an image
def predict_sign(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Function to handle the camera feed
def camera_stream():
    global cap, current_word, start_time, sentence, displayed_word, last_sentence, running

    if running:
        ret, frame = cap.read()
        if ret:
            # Process the frame using MediaPipe Holistic to get hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # If hand landmarks are detected, predict the sign
            if results.left_hand_landmarks or results.right_hand_landmarks:
                predicted_sign = predict_sign(frame, model)

                if predicted_sign == current_word:
                    if time.time() - start_time >= 0.75:
                        if len(sentence) == 0 or sentence.split()[-1] != predicted_sign:
                            sentence += f"{predicted_sign} "
                        displayed_word = predicted_sign
                        current_word = None
                else:
                    current_word = predicted_sign
                    start_time = time.time()

                prediction_var.set(f'Predicted Word: {predicted_sign}')
            else:
                if sentence:
                    # No grammar correction now, just clear the sentence
                    last_sentence = sentence.strip()
                    sentence = ""
                    last_sentence_var.set(f'Last Sentence: {last_sentence}')

            sentence_var.set(f'Sentence: {sentence}')

            # Convert the image for tkinter and update the label
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)  # Ensure this is not commented out

        # Use root.after to schedule the next frame update
        root.after(30, camera_stream)  # Update every 30 ms for smooth performance

# Function to start the camera in a separate thread
def start_camera():
    global cap, running
    if not running:
        # Initialize the camera only if it's not already running
        cap = cv2.VideoCapture(0)
        
        # Check if the camera is opened successfully
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        running = True
        camera_stream()  # Start the camera feed directly in the main thread

# Function to stop the camera
def stop_camera():
    global running, cap
    if running:
        running = False
        if cap is not None:
            cap.release()
            cap = None

# Function to preprocess and process text for videos
def process_text(text):
    text = text.lower()
    filtered_text = []

    # Tokenize and process text without spacy
    words = text.split()
    for word in words:
        # Assume all words are valid for now
        filtered_text.append(word)

    processed_words = []
    for w in filtered_text:
        path = os.path.join('static', 'words', f'{w}.mp4')
        if not os.path.exists(path):
            processed_words.extend(list(w))
        else:
            processed_words.append(w)

    return processed_words

# Function to play video files
def play_video(idx):
    global is_paused, current_word_idx, video_paths, video_thread

    if idx >= len(video_paths) or is_paused:
        return

    video_path = video_paths[idx]
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and not is_paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Use after to schedule the update
        root.after(0, update_image, imgtk)
        
        time.sleep(0.03)  # Adjust delay as necessary

    cap.release()
    if not is_paused:
        play_video(idx + 1)  # Play next video in the list

def update_image(imgtk):
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

def start_video_thread(start_idx=0):
    global video_thread
    video_thread = threading.Thread(target=play_video, args=(start_idx,))
    video_thread.start()

def toggle_play_pause():
    global is_paused
    is_paused = not is_paused
    play_pause_button.config(text="Play" if is_paused else "Pause")

def play_selected_word():
    global current_word_idx, is_paused
    selected_idx = listbox.curselection()
    if selected_idx:
        is_paused = False
        current_word_idx = selected_idx[0]
        start_video_thread(current_word_idx)

def generate_animation():
    global current_word_idx, words_list, video_paths
    text = text_input.get()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    words_list = process_text(text)
    video_paths = [os.path.join('static', 'words', f'{word}.mp4') for word in words_list]
    listbox.delete(0, tk.END)
    for word in words_list:
        listbox.insert(tk.END, word)
    current_word_idx = 0
    start_video_thread()

# Function to reset video and camera
def reset_video():
    global current_word_idx, video_paths, words_list, is_paused, running, cap

    if video_thread and video_thread.is_alive():
        is_paused = True
        video_thread.join()

    listbox.delete(0, tk.END)

    current_word_idx = 0
    video_paths = []
    words_list = []

    sentence_var.set('Sentence: ')
    last_sentence_var.set('Last Sentence: ')

    if running:
        stop_camera()

# Create the main application window
root = tk.Tk()
root.title("Sign Language Conversion")
root.geometry("1200x700")  # Adjusted size for better space distribution
root.configure(bg="#e6ecff")  # Soft gradient-like color

# Heading
title_label = ttk.Label(root, text="Sign Language Conversion", font=("Arial", 28, "bold"), background="#e6ecff")
title_label.pack(pady=10)

# Main container frame
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Configure grid weights for fixed 50-50 distribution
main_frame.grid_columnconfigure(0, weight=1, uniform="equal")  # Left 50%
main_frame.grid_columnconfigure(1, weight=1, uniform="equal")  # Right 50%
main_frame.grid_rowconfigure(0, weight=1)

# Left Frame for Sign to Text
left_frame = ttk.LabelFrame(main_frame, text="Sign to Text", padding=(10, 10), style="LeftFrame.TLabelframe")
left_frame.grid(row=0, column=0, padx=14, pady=10, sticky="nsew")

# Right Frame for Text to Animation
right_frame = ttk.LabelFrame(main_frame, text="Text to Animation", padding=(10, 10), style="RightFrame.TLabelframe")
right_frame.grid(row=0, column=1, padx=14, pady=10, sticky="nsew")

# Sign to Text Section (Left)
camera_frame = ttk.Frame(left_frame)
camera_frame.pack(fill=tk.BOTH, expand=True)

camera_label = tk.Label(camera_frame)  # Changed to tk.Label
camera_label.pack(fill=tk.BOTH, expand=True) 

prediction_var = tk.StringVar()
prediction_label = ttk.Label(left_frame, textvariable=prediction_var, font=("Arial", 14), background="#e1f5fe")
prediction_label.pack(pady=2, fill=tk.X, anchor='w')

sentence_var = tk.StringVar()
sentence_label = ttk.Label(left_frame, textvariable=sentence_var, font=("Arial", 14), background="#e1f5fe")
sentence_label.pack(pady=2, fill=tk.X, anchor='w')

last_sentence_var = tk.StringVar()
last_sentence_label = ttk.Label(left_frame, textvariable=last_sentence_var, font=("Arial", 14), background="#e1f5fe")
last_sentence_label.pack(pady=2, fill=tk.X, anchor='w')

# Text to Animation Section (Right)
video_frame = ttk.Frame(right_frame)
video_frame.pack(fill=tk.BOTH, expand=True)  # Expands frame in both directions

# Create video label
video_label = tk.Label(video_frame)  # Changed to tk.Label
video_label.pack(expand=True)

# Controls and Input
controls_frame = ttk.Frame(right_frame, padding=(10, 10), style="ControlsFrame.TFrame")
controls_frame.pack(pady=10, fill=tk.X)

text_input = tk.Entry(controls_frame, font=("Arial", 14), width=50, bd=2, relief="solid")
text_input.pack(side=tk.TOP, padx=7, pady=2, fill=tk.X)

buttons_frame = ttk.Frame(controls_frame)
buttons_frame.pack(side=tk.TOP, pady=2)

# Add buttons with black text and hover effect
play_pause_button = tk.Button(buttons_frame, text="Pause", command=toggle_play_pause, font=("Arial", 14),
                              bg="red", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                              relief="raised", padx=5, pady=2)
play_pause_button.pack(side=tk.LEFT, padx=10)

play_word_button = tk.Button(buttons_frame, text="Play Selected Word", command=play_selected_word, font=("Arial", 14),
                             bg="aqua", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                             relief="raised", padx=5, pady=2)
play_word_button.pack(side=tk.LEFT, padx=10)

generate_animation_button = tk.Button(buttons_frame, text="Generate Animation", command=generate_animation, font=("Arial", 14),
                                      bg="#4CAF50", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                                      relief="raised", padx=5, pady=2)
generate_animation_button.pack(side=tk.LEFT, padx=10)

reset_button = tk.Button(buttons_frame, text="Reset", command=reset_video, font=("Arial", 14),
                         bg="red", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                         relief="raised", padx=5, pady=2)
reset_button.pack(side=tk.LEFT, padx=10)

# Listbox for words
listbox_frame = ttk.LabelFrame(right_frame, text="Word List", padding=(2, 2), style="ListboxFrame.TLabelframe")
listbox_frame.pack(padx=2, pady=2)  # Padding around the frame is reduced to 2 pixels

# Create Listbox with fixed size: 3 rows, 30 characters wide
listbox = tk.Listbox(listbox_frame, font=("Arial", 14), bd=2, relief="solid", width=30, height=3)
listbox.pack(padx=2, pady=2)  # Padding around the Listbox is reduced to 2 pixels

# Start/Stop Camera Buttons at bottom of root
button_frame = ttk.Frame(root, padding=(10, 10))
button_frame.pack(fill=tk.X, padx=10)

start_button = tk.Button(button_frame, text="Start Camera", command=start_camera, font=("Arial", 14),
                         bg="#28ee04", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                         relief="raised", padx=5, pady=5)
start_button.pack(side=tk.LEFT, padx=5)

stop_button = tk.Button(button_frame, text="Stop Camera", command=stop_camera, font=("Arial", 14),
                        bg="#d80202", fg="black", bd=1, highlightthickness=1, highlightbackground="#333",
                        relief="raised", padx=5, pady=5)
stop_button.pack(side=tk.LEFT, padx=5)

# Updated styles
style = ttk.Style()
style.configure("LeftFrame.TLabelframe",
                background="#bbdefb",  # Light blue background
                font=("Arial", 14, "bold"))
style.configure("RightFrame.TLabelframe",
                background="#c8e6c9",  # Light green background
                font=("Arial", 14, "bold"))
style.configure("ControlsFrame.TFrame",
                background="#f1f8e9")  # Light yellow background
style.configure("ListboxFrame.TLabelframe",
                background="#ffe0b2",  # Light orange background
                font=("Arial", 14, "bold"))

# Start the Tkinter main loop
root.mainloop()
