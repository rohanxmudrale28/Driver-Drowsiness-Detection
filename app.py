import cv2
import time
import threading
import simpleaudio as sa
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os 


# Haar cascades initialization 
# NOTE: These XML files MUST be in the same folder as this script
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# IMPORTANT CHECK: Ensure all three files loaded correctly.
if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
    print("FATAL ERROR: One or more Haar Cascade XML files failed to load.")
    print("Please ensure the three XML files are in the same directory as this script.")
    exit()

# Global alert state for sound
alert_playing = False

def alert_sound():
    """Plays an alarm sound in a separate thread."""
    global alert_playing
    if alert_playing:
        return
    alert_playing = True

    def play_alarm():
        global alert_playing
        try:
            # NOTE: Place your 'alarm.wav' file in the project directory
            wave_obj = sa.WaveObject.from_wave_file("alarm.wav") 
            play_obj = wave_obj.play()
            # The alert sound will play even if the fuzzy score drops, which is a simple approach.
            # For real-world use, you might use a shorter sound or add logic to stop playback.
            play_obj.wait_done() 
        except FileNotFoundError:
            print("ERROR: Alarm sound file not found. Ensure 'alarm.wav' is in the project folder.")
        except Exception as e:
            print(f"ERROR playing sound: {e}")
        finally:
            alert_playing = False

    threading.Thread(target=play_alarm, daemon=True).start()


# Fuzzy logic inputs (Antecedents)
blink_duration = ctrl.Antecedent(np.arange(0, 10, 0.1), 'blink_duration')
eye_state = ctrl.Antecedent(np.arange(0, 2, 1), 'eye_state') # 0=Closed, 1=Open
yawn_count = ctrl.Antecedent(np.arange(0, 10, 1), 'yawn_count')

# Fuzzy logic output (Consequent)
sleep_risk = ctrl.Consequent(np.arange(0, 100, 1), 'sleep_risk')

# Membership functions 
blink_duration['short'] = fuzz.trapmf(blink_duration.universe, [0, 0, 1, 2])
blink_duration['medium'] = fuzz.trimf(blink_duration.universe, [1, 3, 5])
blink_duration['long'] = fuzz.trapmf(blink_duration.universe, [4, 6, 10, 10])

eye_state['closed'] = fuzz.trimf(eye_state.universe, [0, 0, 0.5])
eye_state['open'] = fuzz.trimf(eye_state.universe, [0.5, 1, 1])

yawn_count['few'] = fuzz.trapmf(yawn_count.universe, [0, 0, 2, 3])
yawn_count['moderate'] = fuzz.trimf(yawn_count.universe, [2, 4, 6])
yawn_count['many'] = fuzz.trapmf(yawn_count.universe, [5, 7, 10, 10])

sleep_risk['low'] = fuzz.trimf(sleep_risk.universe, [0, 0, 40])
sleep_risk['medium'] = fuzz.trimf(sleep_risk.universe, [30, 50, 70])
sleep_risk['high'] = fuzz.trimf(sleep_risk.universe, [60, 85, 100])


# Combined Rules for Sleep Risk
BD_L = blink_duration['long']
BD_M = blink_duration['medium']
BD_S = blink_duration['short']
ES_C = eye_state['closed']
ES_O = eye_state['open']
YC_M = yawn_count['many']
YC_O = yawn_count['moderate']
YC_F = yawn_count['few']


final_rules = [
    # HIGH RISK
    ctrl.Rule(BD_L & ES_C, sleep_risk['high']),
    ctrl.Rule(YC_M, sleep_risk['high']),
    ctrl.Rule(BD_M & ES_C & YC_O, sleep_risk['high']), 

    # MEDIUM RISK
    ctrl.Rule(BD_M & ES_C, sleep_risk['medium']),
    ctrl.Rule(YC_O & ES_O, sleep_risk['medium']), 
    ctrl.Rule(BD_S & YC_M, sleep_risk['medium']), 
    ctrl.Rule(BD_S & ES_C & YC_F, sleep_risk['medium']),

    # LOW RISK
    ctrl.Rule(BD_S & ES_O & YC_F, sleep_risk['low']),
    ctrl.Rule(ES_O & YC_F, sleep_risk['low']),
]

sleep_risk_ctrl = ctrl.ControlSystem(final_rules)
sleep_risk_sim = ctrl.ControlSystemSimulation(sleep_risk_ctrl)


#Car representation

def main():
    # Robustly try to open the camera
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    except:
        cap = cv2.VideoCapture(0) # Fallback to default
        
    if not cap.isOpened():
        print("Error: Cannot open camera (index 0 failed).")
        return

    # Detection/Counter variables
    eyes_closed_start_time = None
    blink_duration_sec = 0
    yawn_count_val = 0
    yawn_detected = False
    yawn_start_time = None
    yawn_threshold = 1.5

    # Drowsiness Alert Variables
    alert_triggered = False
    DROWSINESS_THRESHOLD = 70 

    # --- SIMULATION VARIABLES ---
    SIM_WIDTH, SIM_HEIGHT = 640, 480
    MAX_SPEED = 100
    MIN_SPEED = 20
    
    # Car State
    car_speed = MAX_SPEED
    lane_position = SIM_WIDTH // 2 # Center of the lane (lateral position)
    
    # Control factors (smoother transitions)
    SPEED_ADJUST_RATE = 1.0 
    LANE_DRIFT_RATE = 5 
    
    print("System Initialized. Monitoring Driver and running Car Simulation.")

    while True:
        if not cap.isOpened():
            print("Camera lost connection. Exiting.")
            break 

        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Flip the frame for a mirror effect, which is often more intuitive for webcam use
        frame = cv2.flip(frame, 1) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lower scaleFactor and increase minNeighbors for more stable face detection
        faces = face_cascade.detectMultiScale(gray, 1.2, 5) 

        eyes_detected = False
        mouth_detected = False

        # --- A. FACE, EYE, MOUTH DETECTION ---
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi_gray = gray[y:y+h, x:x+w]
            
            # Eye Detection
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.05, minNeighbors=2)
            if len(eyes) > 0:
                eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            # Mouth/Yawn Detection - USING IMPROVED PARAMETERS & MAR CHECK
            mouths = mouth_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=7) # Adjusted
            for (mx, my, mw, mh) in mouths:
                if my > h / 2: # Only consider mouths in the lower half of the face
                    # Yawn check: mouth wide enough AND tall enough (MAR check)
                    if mw > 0.3 * w and mh > 0.4 * mw: 
                        mouth_detected = True
                        cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 0, 255), 2)


        # --- B. DROWSINESS METRICS CALCULATION ---

        # Blink Duration/Eyes Closed Time
        if not eyes_detected:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            blink_duration_sec = time.time() - eyes_closed_start_time
        else:
            eyes_closed_start_time = None
            blink_duration_sec = 0
            
        eye_state_val = 1 if eyes_detected else 0 # 1 for open, 0 for closed

        # Yawn Count Logic
        if mouth_detected:
            if yawn_start_time is None:
                yawn_start_time = time.time()
            
            if (time.time() - (yawn_start_time or time.time())) > yawn_threshold:
                yawn_detected = True 
        else:
            if yawn_start_time is not None:
                if yawn_detected:
                    yawn_count_val += 1
                
                yawn_start_time = None
                yawn_detected = False


        # --- C. FUZZY LOGIC COMPUTATION ---
        
        # Clamp inputs to the universe max to avoid boundary errors
        bd_input = min(blink_duration_sec, blink_duration.universe.max()) 
        yc_input = min(yawn_count_val, yawn_count.universe.max())

        # Set inputs
        sleep_risk_sim.input['blink_duration'] = bd_input
        sleep_risk_sim.input['eye_state'] = eye_state_val
        sleep_risk_sim.input['yawn_count'] = yc_input
        
        sleep_risk_score = 0.0 # Default to low risk

        # Safely compute the fuzzy output
        try:
            sleep_risk_sim.compute()
            sleep_risk_score = sleep_risk_sim.output['sleep_risk']
        except Exception:
            sleep_risk_score = 0.0 

        drowsiness_score = sleep_risk_score 

        # --- D. DROWSINESS ALERT LOGIC ---

        if drowsiness_score > DROWSINESS_THRESHOLD and not alert_triggered:
            alert_sound()
            alert_triggered = True
        elif drowsiness_score <= DROWSINESS_THRESHOLD and alert_triggered:
            alert_triggered = False

        
        # --- E. CAR SIMULATION LOGIC AND RENDERING (NEW SECTION) ---
        
        # 1. Update Car State based on Drowsiness Score
        
        if drowsiness_score > DROWSINESS_THRESHOLD:
            # High Risk: SLOW DOWN and DRIFT
            
            # Slow down toward MIN_SPEED 
            if car_speed > MIN_SPEED:
                car_speed -= SPEED_ADJUST_RATE
            
            # Drift out of lane (towards the right edge)
            lane_position += LANE_DRIFT_RATE
            
        else:
            # Low/Medium Risk: SPEED UP and CENTER
            
            # Speed up toward MAX_SPEED
            if car_speed < MAX_SPEED:
                car_speed += SPEED_ADJUST_RATE
            
            # Center the car 
            center = SIM_WIDTH // 2
            if abs(lane_position - center) > LANE_DRIFT_RATE:
                if lane_position > center:
                    lane_position -= LANE_DRIFT_RATE
                else:
                    lane_position += LANE_DRIFT_RATE
            else:
                lane_position = center
        
        # Clamp values to safe boundaries
        car_speed = max(MIN_SPEED, min(MAX_SPEED, car_speed))
        lane_position = max(100, min(SIM_WIDTH - 100, lane_position)) # Keep car visible

        
        # 2. Render the Simulation Window
        
        # Create a blank black image for the simulation (The Road)
        sim_frame = np.zeros((SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
        
        # Draw the Road and Lane Markers
        ROAD_COLOR = (100, 100, 100) # Gray
        LANE_COLOR = (255, 255, 255) # White
        # Draw background road
        cv2.rectangle(sim_frame, (0, SIM_HEIGHT // 2), (SIM_WIDTH, SIM_HEIGHT), ROAD_COLOR, -1)
        # Draw center lane marker
        cv2.line(sim_frame, (SIM_WIDTH // 2, SIM_HEIGHT // 2), (SIM_WIDTH // 2, SIM_HEIGHT), LANE_COLOR, 3)

        # Draw the Car - IMPROVED VISUALS
        CAR_COLOR_BODY = (0, 165, 255) # Orange for normal, changes to red if high risk
        CAR_COLOR_WINDOW = (100, 100, 200) # Light blue for windows
        CAR_COLOR_WHEEL = (50, 50, 50) # Dark gray for wheels
        CAR_COLOR_HIGHLIGHT = (200, 200, 200) # Light gray for highlights
        
        # If high drowsiness, make the car red
        if drowsiness_score > DROWSINESS_THRESHOLD:
            CAR_COLOR_BODY = (0, 0, 255) # Red

        CAR_WIDTH, CAR_HEIGHT = 80, 100
        
        # Base position of the car (bottom-center)
        car_bottom_center_x = int(lane_position)
        car_bottom_y = SIM_HEIGHT - 20 # 20 pixels from the bottom edge of the frame

        # Body of the car (main rectangle)
        body_width = CAR_WIDTH
        body_height = int(CAR_HEIGHT * 0.7)
        body_top_left = (car_bottom_center_x - body_width // 2, car_bottom_y - body_height)
        body_bottom_right = (car_bottom_center_x + body_width // 2, car_bottom_y)
        cv2.rectangle(sim_frame, body_top_left, body_bottom_right, CAR_COLOR_BODY, -1)
        
        # Cabin/Roof of the car (smaller rectangle on top of the body)
        cabin_width = int(CAR_WIDTH * 0.7)
        cabin_height = int(CAR_HEIGHT * 0.4)
        cabin_top_left = (car_bottom_center_x - cabin_width // 2, body_top_left[1] - cabin_height + 5) # +5 for overlap
        cabin_bottom_right = (car_bottom_center_x + cabin_width // 2, body_top_left[1] + 5)
        cv2.rectangle(sim_frame, cabin_top_left, cabin_bottom_right, CAR_COLOR_HIGHLIGHT, -1) # Roof color
        
        # Windows
        window_height = int(cabin_height * 0.6)
        # Front window
        cv2.rectangle(sim_frame, (cabin_top_left[0] + 5, cabin_top_left[1] + 5), 
                      (cabin_bottom_right[0] - 5, cabin_top_left[1] + window_height), CAR_COLOR_WINDOW, -1)
        # Back window
        cv2.rectangle(sim_frame, (cabin_top_left[0] + 5, cabin_bottom_right[1] - window_height - 5), 
                      (cabin_bottom_right[0] - 5, cabin_bottom_right[1] - 5), CAR_COLOR_WINDOW, -1)

        # Wheels (circles)
        wheel_radius = int(CAR_WIDTH * 0.15)
        wheel_y = car_bottom_y - wheel_radius // 2
        # Front wheel
        cv2.circle(sim_frame, (car_bottom_center_x - body_width // 2 + wheel_radius, wheel_y), wheel_radius, CAR_COLOR_WHEEL, -1)
        # Rear wheel
        cv2.circle(sim_frame, (car_bottom_center_x + body_width // 2 - wheel_radius, wheel_y), wheel_radius, CAR_COLOR_WHEEL, -1)

        # Headlights (small white squares at the front)
        headlight_width = int(CAR_WIDTH * 0.1)
        headlight_height = int(CAR_WIDTH * 0.08)
        
        cv2.rectangle(sim_frame, (car_bottom_center_x - body_width // 2 - headlight_width + 5, body_top_left[1] + 10),
                      (car_bottom_center_x - body_width // 2 + 5, body_top_left[1] + 10 + headlight_height), (255, 255, 255), -1)
        cv2.rectangle(sim_frame, (car_bottom_center_x + body_width // 2 - 5 - headlight_width, body_top_left[1] + 10),
                      (car_bottom_center_x + body_width // 2 - 5, body_top_left[1] + 10 + headlight_height), (255, 255, 255), -1)

        
        # Draw Speedometer
        cv2.putText(sim_frame, f'SPEED: {car_speed:.0f} MPH', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw Lane Status
        if abs(lane_position - SIM_WIDTH // 2) > 100:
             cv2.putText(sim_frame, '!!! CRITICAL DRIFT !!!', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif abs(lane_position - SIM_WIDTH // 2) > 50:
             cv2.putText(sim_frame, 'Drifting', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


        
        risk_color = (0, 0, 255) if drowsiness_score > DROWSINESS_THRESHOLD else (0, 255, 0)
        
        # Display Drowsiness Info on the camera frame
        cv2.putText(frame, '--- Drowsiness Detector ---', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Risk Score: {drowsiness_score:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        cv2.putText(frame, f'Blink Time: {blink_duration_sec:.1f}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Yawn Count: {yawn_count_val}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if drowsiness_score > DROWSINESS_THRESHOLD:
            cv2.putText(frame, '!!! HIGH RISK - WAKE UP !!!', (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif drowsiness_score > DROWSINESS_THRESHOLD - 20: 
            cv2.putText(frame, 'Tiredness Detected', (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


        cv2.imshow('1. Driver Monitor (Webcam)', frame)
        cv2.imshow('2. Car Simulation (Output)', sim_frame) # Show the simulation

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # Crucial step: Ensure all resources are released properly
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
