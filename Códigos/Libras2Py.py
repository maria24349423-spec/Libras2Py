import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3  # Biblioteca para conversão de texto em fala
import threading  # Biblioteca para executar a fala em uma thread separada

# Inicializar o mecanismo de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidade da fala

# Função para falar em uma thread separada
def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/Wwbbnn220801/Downloads/blablabla/Model/keras_model.h5", "C:/Users/Wwbbnn220801/Downloads/blablabla/Model/labels.txt")
offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

# Variáveis para controle de tempo
prediction_time = 2  # Intervalo de 2 segundos
last_prediction_time = time.time()  # Armazena o último tempo de previsão
hand_visible_time = 3  # Tempo que a mão deve estar visível antes da previsão
hand_detected_time = None  # Tempo em que a mão foi detectada

while True:
    success, img = cap.read()
    imgOutput = img.copy()  # Mantém a imagem original da câmera
    hands, img = detector.findHands(img)

    if hands:
        # Mão detectada
        if hand_detected_time is None:
            hand_detected_time = time.time()  # Armazena o tempo em que a mão foi detectada
        else:
            # Verifica se a mão está visível por mais de hand_visible_time
            if time.time() - hand_detected_time >= hand_visible_time:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

                # Inverter a parte do imgCrop
                imgCrop = cv2.flip(imgCrop, 1)  # Inverter a imagem do recorte da mão
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                # Verificar se o intervalo de tempo foi atingido para fazer a previsão
                current_time = time.time()
                if current_time - last_prediction_time >= prediction_time:
                    # Atualizar o tempo da última previsão
                    last_prediction_time = current_time

                    # Fazer a previsão
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                    # Obter o texto correspondente à previsão
                    prediction_text = labels[index]

                    # Iniciar a fala em uma nova thread para evitar bloqueios
                    threading.Thread(target=speak, args=(prediction_text,)).start()

                    # Exibir a previsão na imagem
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgOutput, prediction_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                cv2.imshow('ImageCrop', imgCrop)  # Exibe a imagem da mão invertida
                cv2.imshow('ImageWhite', imgWhite)

    else:
        # Se nenhuma mão foi detectada, resetar o tempo de detecção
        hand_detected_time = None  # Redefine o tempo se a mão não estiver visível

    cv2.imshow('Image', imgOutput)  # Exibe a imagem original da câmera
    cv2.waitKey(1)

# Liberar a webcam e fechar janelas
cap.release()
cv2.destroyAllWindows()
