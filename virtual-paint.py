import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Constants and initializations
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
pen_size = 5
drawing = False
undo_stack = []
redo_stack = []
colorIndex = 0

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

paintWindow = np.zeros((471, 636, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            center = (landmarks[8][0], landmarks[8][1])
            thumb = (landmarks[4][0], landmarks[4][1])

            if thumb[1] - center[1] < 30:
                bpoints.append(deque(maxlen=512))
                gpoints.append(deque(maxlen=512))
                rpoints.append(deque(maxlen=512))
                ypoints.append(deque(maxlen=512))

            elif center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                # Add conditions for other color buttons

            else:
                if colorIndex == 0:
                    bpoints[-1].appendleft(center)
                elif colorIndex == 1:
                    gpoints[-1].appendleft(center)
                elif colorIndex == 2:
                    rpoints[-1].appendleft(center)
                elif colorIndex == 3:
                    ypoints[-1].appendleft(center)

    # Draw lines
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
