import torch
from ultralytics import YOLO
import cv2
import pyautogui
import math
import time

pyautogui.FAILSAFE = False

device = torch.device('cpu')

model = YOLO('D:/ai-project/best1.pt')

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

prev_center_x, prev_center_y = None, None
sensitivity = 2
min_movement = 5

last_click_time = time.time()
click_delay = 0.5  
initial_mouse_x, initial_mouse_y = pyautogui.position()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось захватить кадр")
        break

    frame = cv2.flip(frame, 1)

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            current_time = time.time()
            if class_id == 2:
                pyautogui.mouseDown()
            if class_id == 7:
                pyautogui.mouseUp()
            if class_id == 4:
                if current_time - last_click_time > click_delay:
                    pyautogui.hotkey('alt', 'f4')
                    last_click_time = current_time 

            if class_id == 12:
                if current_time - last_click_time > click_delay:
                    pyautogui.rightClick()
                    last_click_time = current_time

            if class_id == 19:
                if current_time - last_click_time > click_delay:
                    pyautogui.doubleClick()
                    last_click_time = current_time

            if class_id == 3:
                if prev_center_x is None or prev_center_y is None:
                    prev_center_x, prev_center_y = center_x, center_y

                delta_x = (center_x - prev_center_x) * sensitivity
                delta_y = (center_y - prev_center_y) * sensitivity

                movement_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

                if movement_distance > min_movement:
                    current_mouse_x, current_mouse_y = pyautogui.position()

                    new_x = min(max(current_mouse_x + delta_x, 0), pyautogui.size().width - 1)
                    new_y = min(max(current_mouse_y + delta_y, 0), pyautogui.size().height - 1)

                    pyautogui.moveTo(new_x, new_y, duration=0)

                prev_center_x, prev_center_y = center_x, center_y

