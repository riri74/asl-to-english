# creates new csv file asl_features with all required data

import os
import cv2
import mediapipe as mp
import pandas as pd

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Paths
input_dir = "asl_dataset"  # Replace with the path to your preprocessed images
output_csv = "asl_features.csv"  # CSV file to save the extracted features

# Initialize list to store features
features_list = []

# Process each class (subfolder)
for label in os.listdir(input_dir):
    class_path = os.path.join(input_dir, label)
    if os.path.isdir(class_path):  # Check if it's a folder
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Read the image
                image = cv2.imread(img_path)

                if image is not None:
                    # Convert to RGB (required by Mediapipe)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect hand landmarks
                    results = hands.process(image_rgb)

                    # Check if any hands are detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract 21 landmarks (x, y, z) and flatten into a single list
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])

                            # Append features with the label
                            features_list.append([file, label] + landmarks)
                    else:
                        print(f"No hands detected in: {img_path}")
                else:
                    print(f"Failed to load image: {img_path}")

# Close Mediapipe
hands.close()

# Create a DataFrame
columns = ['filename', 'label'] + [f'lm_{i}_{axis}' for i in range(21) for axis in ('x', 'y', 'z')]
df = pd.DataFrame(features_list, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Feature extraction complete! Saved to {output_csv}")
