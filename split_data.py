import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import mediapipe as mp
import os

# ================== CONFIG ==================
MODEL_PATH = "hand_sign_model.pth"
CLASS_NAME_PATH = "class_names.json"
IMG_SIZE = 28
CONF_THRESHOLD = 0.75   # only show result if confident
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== LOAD CLASSES ==================
if not os.path.exists(CLASS_NAME_PATH):
    raise RuntimeError("Missing class_names.json")

with open(CLASS_NAME_PATH, "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
print("Classes:", class_names)
print("Device:", DEVICE)

# ================== MODEL ==================
class HandSignNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ================== LOAD MODEL ==================
model = HandSignNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded OK")

# ================== PREPROCESS ==================
def preprocess(gray_roi):
    # resize đúng như lúc train
    img = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # normalize giống train
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    # (1,1,28,28)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================== WEBCAM ==================
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # bounding box
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # padding lớn để giống dataset
            pad = 80
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # square crop
            bw = x2 - x1
            bh = y2 - y1
            if bw > bh:
                diff = (bw - bh) // 2
                y1 = max(0, y1 - diff)
                y2 = min(h, y2 + diff)
            else:
                diff = (bh - bw) // 2
                x1 = max(0, x1 - diff)
                x2 = min(w, x2 + diff)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            inp = preprocess(gray)

            with torch.no_grad():
                out = model(inp)
                prob = F.softmax(out, dim=1)
                conf, pred = torch.max(prob, 1)

            if conf.item() > CONF_THRESHOLD:
                label = class_names[pred.item()]
                text = f"{label} {conf.item():.0%}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("ROI", gray)

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
