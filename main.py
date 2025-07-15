import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define the box/window parameters
box_x, box_y = 300, 200
box_size = 100
dragging = False
prev_pos = None

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]  # Index finger tip
            x, y = int(lm.x * w), int(lm.y * h)

            # Check if the finger is over the box
            if box_x < x < box_x + box_size and box_y < y < box_y + box_size:
                if cv2.waitKey(1) & 0xFF == ord('d'):  # Optional: Use 'd' key to simulate pinch
                    dragging = True

            if dragging:
                box_x = x - box_size // 2
                box_y = y - box_size // 2

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    else:
        dragging = False  # Stop dragging if hand not visible

    # Draw the box
    cv2.rectangle(img, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Moving Window", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
