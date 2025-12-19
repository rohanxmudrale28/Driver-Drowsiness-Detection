# Driver-Drowsiness-Detection
# 🚗 Driver Drowsiness Detection via Fuzzy Logic & Computer Vision

This project implements a sophisticated Driver Assistance System (ADAS) that uses **Fuzzy Logic** to monitor driver fatigue in real-time. Unlike traditional binary systems, this project considers drowsiness as a spectrum, adjusting car behavior (speed and lane position) gradually based on a calculated Risk Score.



## 🌟 Key Features
- **Non-Binary Detection:** Uses Fuzzy Logic to calculate a "Sleep Risk" score (0-100) instead of a simple "awake/asleep" toggle.
- **Multi-Factor Input:** Monitors Blink Duration, Eye State, and Yawn Count simultaneously using OpenCV.
- **Buffer Logic:** Includes a "buffer time" to avoid false positives (e.g., when a driver looks down at the dashboard).
- **Automated Car Simulation:** - **High Risk:** The car gradually slows down and drifts safely toward the road shoulder.
  - **Recovery:** Once the driver wakes up (detected via webcam), the car simulates a return to the center of the lane and resumes speed.
- **Audio Alerts:** Integrated threading for real-time buzzer alarms.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** - `OpenCV`: For real-time video processing and Haar Cascade detections.
  - `scikit-fuzzy`: To implement the Fuzzy Inference System (FIS).
  - `NumPy`: For numerical operations and frame simulation.
  - `Simpleaudio`: For low-latency alarm sounds.

## 🧠 Fuzzy Logic Design
The system uses the following **Antecedents (Inputs)**:
1. **Blink Duration:** Short, Medium, Long.
2. **Eye State:** Open or Closed.
3. **Yawn Count:** Few, Moderate, Many.

**Consequent (Output):**
- **Sleep Risk:** Low, Medium, High.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Webcam

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Driver-Drowsiness-Fuzzy-Logic.git](https://github.com/YourUsername/Driver-Drowsiness-Fuzzy-Logic.git)
