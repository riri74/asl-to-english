import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib

# load the model
model = joblib.load('svc_model.pkl')

# mp setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# open webcam
cap = cv.VideoCapture(0)

while True:
    success, frame = cap.read()

    # flip frame for mirror view
    frame = cv.flip(frame, 1)

    # converting img to rgb since cv2 follows bgr scheme
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # collect landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # convert to numpy arr and preprocess
            landmarks = np.array(landmarks).reshape(1, -1)

            # make prediction
            pred = model.predict(landmarks)
            pred_label = pred[0]

        # draw landmarks and prediction on screen
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
        cv.putText(frame, f'Prediction: {pred_label}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)

    
    cv.imshow('ASL recognition', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
