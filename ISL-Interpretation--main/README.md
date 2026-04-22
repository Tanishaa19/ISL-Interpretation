# ISL (Indian Sign Language) Prediction Model

## 📌 Project Description
This project focuses on real-time recognition of Indian Sign Language (ISL) gestures using Computer Vision and Machine Learning techniques. It captures hand movements through a webcam and predicts the corresponding sign instantly.

## ✨ Features
- Real-time hand detection using MediaPipe  
- Machine Learning based gesture prediction  
- Supports single and multiple hand detection  
- Text-to-Speech (TTS) for predicted output  
- Voice-to-Sign feature (speech input → sign display)  
- Smooth predictions using frame buffering  

## 🛠 Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy  
- SpeechRecognition  

## 📁 Project Structure
ISL-Interpretation/
│── 01_extract_landmarks.py
│── 02_train_model.py
│── 03_real_time_prediction.py
│── requirements.txt
│── README.md
│── models/
│── landmarks_data/

## 🚀 How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Execute the pipeline:
python 01_extract_landmarks.py
python 02_train_model.py
python 03_real_time_prediction.py

## 🎮 Controls
- q → Quit application  
- s → Capture screenshot  
- v → Activate voice input mode  

## 🧠 How It Works
1. MediaPipe detects 21 hand landmarks  
2. Feature vectors are extracted from hand coordinates  
3. Machine learning model is trained on dataset  
4. Real-time webcam input is used for prediction  
5. Output is displayed as text and audio  

## 📂 Dataset
The dataset includes images of ISL gestures such as:
- Hello  
- Yes  
- Stop  
- Water  
- Pray  

## ⚠️ Note
This project is developed for academic and learning purposes.

## 📌 Future Improvements
- Increase dataset size for better accuracy  
- Add more gesture classes   
- Deploy as a web or mobile application  

## ⭐ Conclusion
This project demonstrates how AI and Computer Vision can be applied to improve communication through sign language recognition.
