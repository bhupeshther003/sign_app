import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, Response, jsonify, request, jsonify
import time
import os
import spacy
import pickle


# Initialize camera and other variables
cap = None
current_prediction = 'None'
sentence = ""
last_sentence = ""
start_time = None
previous_word = None
last_detection_time = None
SENTENCE_TIMEOUT = 0.60  # seconds to consider a word continuous
INACTIVITY_TIMEOUT = 2.0  # seconds to wait before updating last sentence when no hand is detected

nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
cap = None
# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)




# Labels dictionary for prediction
labels_dict = {0: 'afternoon', 1: 'art', 2: 'banana', 3: 'believe', 4: 'birthday', 5: 'buy', 6: 'calculator', 7: 'camera', 8: 'can', 9: 'cat', 10: 'check', 11: 'clean', 12: 'country', 13: 'dad', 14: 'document', 15: 'early', 16: 'eat', 17: 'environment', 18: 'every', 19: 'fish', 20: 'happy', 21: 'how', 22: 'i', 23: 'in', 24: 'independent', 25: 'india', 26: 'keep', 27: 'kid', 28: 'like', 29: 'live', 30: 'many', 31: 'me', 32: 'meeting', 33: 'morning', 34: 'movie', 35: 'my', 36: 'night', 37: 'now', 38: 'on', 39: 'people', 40: 'ready', 41: 'swim', 42: 'to', 43: 'try', 44: 'walk', 45: 'watch', 46: 'weekend', 47: 'work', 48: 'yes', 49: 'you', 50: 'yourself'}
# Initialize video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    global current_prediction, sentence, last_sentence, previous_word, start_time, previous_added_word

    while True:
        success, frame = cap.read()
        if not success:
            break

        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = "Unknown"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                expected_features = 84
                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))
                elif len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                # Update current prediction
                if isinstance(predicted_character, str):
                    current_prediction = predicted_character if predicted_character in labels_dict.values() else "Unknown"
                else:
                    current_prediction = labels_dict.get(predicted_character, "Unknown")

                # Handle sentence logic
                current_time = time.time()
                if current_prediction != "Unknown":
                    # Add space if there's already a sentence
                    if sentence and previous_added_word:
                        sentence += " "

                    if current_prediction != previous_word:
                        # New word detected
                        start_time = current_time
                        previous_word = current_prediction

                    # Check for continuous prediction
                    if start_time and (current_time - start_time >= SENTENCE_TIMEOUT):
                        # Add word if it's new or if it’s the same word but was not added immediately after
                        if (current_prediction != previous_added_word):
                            sentence += current_prediction
                            previous_added_word = current_prediction  # Update last added word
                            print(f"Updated Sentence: {sentence.strip()}")  # Debug statement

                        # Reset start_time for continuous same word
                        start_time = current_time  # Reset start_time

                # Draw prediction text on the frame
                # cv2.putText(frame, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            # If no hand detected, save current sentence as last sentence
            if sentence:
                last_sentence = sentence.strip()
                print(f"Last Sentence: {last_sentence}")  # Debug statement
            sentence = ""  # Reset current sentence
            previous_word = None  # Reset previous word
            previous_added_word = None  # Reset last added word
            start_time = None  # Reset start time

        # Draw the last sentence on the frame
        # cv2.putText(frame, f"Last Sentence: {last_sentence}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/current_prediction')
def current_prediction_route():
    return jsonify({'prediction': current_prediction, 'sentence': sentence, 'last_sentence': last_sentence})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Release any existing camera
    cap = cv2.VideoCapture(0)  # Start the camera
    if cap.isOpened():
        print("Camera started successfully")
        return jsonify({'message': 'Camera started'}), 200
    else:
        print("Cannot open camera")
        cap = None  # Set cap to None if it fails to open
        return jsonify({'message': 'Cannot open camera'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        if cap.isOpened():
            cap.release()  # Release the camera
        cap = None  # Set cap to None
        print("Camera stopped")
        return jsonify({'message': 'Camera stopped'}), 200
    print("Camera is not running")
    return jsonify({'message': 'Camera is not running'}), 400




#   text to aniamation  genrator code 

@app.route('/animation', methods=['GET', 'POST'])
def animation_view():
    if request.method == 'POST':
        text = request.form.get('sen', '').lower()

        # Process the text using spaCy
        doc = nlp(text)

        # Initialize tense counters
        tense = {
            "future": 0,
            "present": 0,
            "past": 0,
            "present_continuous": 0
        }

        filtered_text = []
        for token in doc:
            pos = token.pos_
            tag = token.tag_

            # Count tenses
            if tag == "MD":
                tense["future"] += 1
            elif pos in ["VERB", "AUX"]:
                if tag in ["VBG", "VBN", "VBD"]:
                    tense["past"] += 1
                elif tag == "VBG":
                    tense["present_continuous"] += 1
                else:
                    tense["present"] += 1

            # Lemmatization
            if pos in ["VERB", "NOUN"]:
                filtered_text.append(token.lemma_)
            elif pos in ["ADJ", "ADV"]:
                filtered_text.append(token.lemma_)
            else:
                filtered_text.append(token.text)

        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            filtered_text = ["Before"] + filtered_text
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in filtered_text:
                filtered_text = ["Will"] + filtered_text
        elif probable_tense == "present":
            if tense["present_continuous"] >= 1:
                filtered_text = ["Now"] + filtered_text

        # Handle static files
        processed_words = []
        for w in filtered_text:
            path = os.path.join(app.static_folder, 'words', f'{w}.mp4')
            if not os.path.exists(path):
                processed_words.extend(list(w))
            else:
                processed_words.append(w)
        filtered_text = processed_words

        return render_template('animation.html', words=filtered_text, text=text)
    else:
        return render_template('animation.html')


@app.route('/coming_soon')
def coming_soon():
    return render_template('coming_soon.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5001 to your desired port number