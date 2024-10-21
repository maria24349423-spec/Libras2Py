import pickle  
import cv2  
import mediapipe as mp  
import numpy as np  
import time  
import pyttsx3  
import threading  

engine = pyttsx3.init()  
engine.setProperty('rate', 150)

def speak(text):  
    engine.say(text)  
    engine.runAndWait()  

model_dict_static = pickle.load(open('./model_estatico.p', 'rb'))  
model_static = model_dict_static['model']  

model_dict_movement = pickle.load(open('./model_movimento.p', 'rb'))  
model_movement = model_dict_movement['model']  

cap = cv2.VideoCapture(0)  

mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles  

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_static_dict = {0: 'A', 1: 'B', 2: 'C'}
labels_movement_dict = {0: 'Oi', 1: 'Obrigado'}

movement_threshold = 0.3  

def calcular_movimento_total(hand_landmarks, width, height):
    movimento_total = 0
    x_centro = width // 2  # Coordenada x do centro da imagem
    y_centro = height // 2  # Coordenada y do centro da imagem

    for lm in hand_landmarks.landmark:
        x = int(lm.x * width)
        y = int(lm.y * height)
        z = lm.z

        deslocamento_x = x - x_centro
        deslocamento_y = y - y_centro
        deslocamento_z = z

        movimento_total += (deslocamento_x ** 2 + deslocamento_y ** 2 + deslocamento_z ** 2) ** 0.5

    return movimento_total


while True:  
    data_static = []  
    data_movement = []  
    x_ = []  
    y_ = []  
    z_ = []  

    ret, frame = cap.read()  
    H, W, _ = frame.shape  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    results = hands.process(frame_rgb)  

    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:  
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i in range(len(hand_landmarks.landmark)):  
                x = hand_landmarks.landmark[i].x  
                y = hand_landmarks.landmark[i].y  
                z = hand_landmarks.landmark[i].z  

                x_.append(x)  
                y_.append(y)  
                z_.append(z)  

            movimento_atual = calcular_movimento_total(hand_landmarks, W, H)  
            data_movement.append(movimento_atual)  
            data_static = np.array([x_[i] for i in range(len(x_))])  

        if len(data_static) != 0:  
            data_static = np.expand_dims(data_static, axis=0)  
            predicted_static = model_static.predict(data_static)  
            predicted_static_label = labels_static_dict[predicted_static[0]]  
            print("Sinal Estático: ", predicted_static_label)  

        if len(data_movement) > 0:  
            data_movement = np.expand_dims(data_movement, axis=0)  
            predicted_movement = model_movement.predict(data_movement)  
            predicted_movement_label = labels_movement_dict[predicted_movement[0]]  

            if predicted_movement_label and max(data_movement[0]) > movement_threshold:  
                print('Sinal Dinâmico:', predicted_movement_label)  
                threading.Thread(target=speak, args=(predicted_movement_label,)).start()  

    cv2.imshow('Sinais de Libras', frame)  

    if cv2.waitKey(10) & 0xFF == 27:  
        break  

cap.release()  
cv2.destroyAllWindows()
