import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import mediapipe as mp 

# --- 1. CONFIG & LOAD ---
if not os.path.exists('class_names.json'):
    print("LỖI: Thiếu file class_names.json!")
    exit()

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL ---
class HandSignNet(nn.Module):
    def __init__(self, num_classes):
        super(HandSignNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = HandSignNet(num_classes).to(DEVICE)
try:
    model.load_state_dict(torch.load("hand_sign_model.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    print(">>> Load model thành công!")
except:
    print("Lỗi load model. Kiểm tra lại file .pth")
    exit()

# --- 3. PREPROCESS ---
def preprocess_input(img_roi):
    # Resize về 28x28
    img_roi = cv2.resize(img_roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # --- DEBUG: Show ảnh model nhìn thấy ---
    # Phóng to lên 200px để mắt thường nhìn thấy được trên màn hình
    debug_view = cv2.resize(img_roi, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Model Input (Debug)', debug_view)
    # ---------------------------------------

    img_roi = img_roi.astype('float32') / 255.0
    img_roi = (img_roi - 0.5) / 0.5 
    tensor = torch.from_numpy(img_roi).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

# --- 4. SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- QUAN TRỌNG: LẬT ẢNH ---
    # Lật ngược lại để Tay Phải của bạn đúng là Tay Phải trong hình
    frame = cv2.flip(frame, 1) 

    h_frame, w_frame, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Tính khung bao
            x_min, y_min = w_frame, h_frame
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w_frame), int(lm.y * h_frame)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Padding rộng ra chút
            pad = 50 
            y_min = max(0, y_min - pad)
            y_max = min(h_frame, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w_frame, x_max + pad)
            
            # Làm vuông khung hình
            box_w = x_max - x_min
            box_h = y_max - y_min
            if box_w > box_h:
                diff = (box_w - box_h) // 2
                y_min = max(0, y_min - diff)
                y_max = min(h_frame, y_max + diff)
            else:
                diff = (box_h - box_w) // 2
                x_min = max(0, x_min - diff)
                x_max = min(w_frame, x_max + diff)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = gray_frame[y_min:y_max, x_min:x_max]
                
                if roi.size > 0:
                    input_tensor = preprocess_input(roi)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                        label = class_names[predicted.item()]
                        score = conf.item()

                    if score > 0.4: # Hạ ngưỡng xuống chút để test
                        cv2.putText(frame, f"{label} {score:.0%}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            except:
                pass

    cv2.imshow('Hand Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()