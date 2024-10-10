import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from collections import deque
import math
import time

# Configurações para detecção de mão com cvzone
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
folder = "C:/Users/Wwbbnn220801/Downloads/blablabla/Imagens/N"

# Configurações para desenhar
points = [deque(maxlen=1024)]
index = 0
kernel = np.ones((5,5),np.uint8)
line_color = (255, 0, 0)

paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Inicializando MediaPipe para desenhar
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Inicializando a webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler cada frame da webcam
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Espelhar o frame verticalmente
    framergb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Adicionando botão de "CLEAR" no frame
    img = cv2.rectangle(img, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.putText(img, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Detectar mãos com MediaPipe para desenho
    result = hands.process(framergb)

    # Detectar mãos com cvzone para captura de imagem
    hands_cvzone, img = detector.findHands(img)
    
    # Processamento para desenho
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Desenhar landmarks no frame
            mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)

        # Obter o dedo indicador para desenhar
        fore_finger = (landmarks[9][0], landmarks[9][1])
        center = fore_finger

        # Verificar se o dedo está sobre o botão CLEAR
        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Botão CLEAR
                points = [deque(maxlen=512)]  # Limpar todas as linhas
                index = 0
                paintWindow[67:,:,:] = 255  # Limpar o canvas
        else:
            points[index].appendleft(center)
    else:
        points.append(deque(maxlen=512))
        index += 1

    # Desenhar no canvas (paintWindow) apenas
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(paintWindow, points[j][k - 1], points[j][k], line_color, 2)

    # Processamento para captura de imagem
    if hands_cvzone:
        hand = hands_cvzone[0]
        x, y, w, h = hand['bbox']

        # Garantir que as coordenadas do recorte estejam dentro dos limites da imagem
        x = max(0, x - offset)
        y = max(0, y - offset)
        w = min(img.shape[1] - x, w + 2 * offset)  # Largura máxima limitada ao tamanho da imagem
        h = min(img.shape[0] - y, h + 2 * offset)  # Altura máxima limitada ao tamanho da imagem
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y:y + h, x:x + w]

        # Verificar se imgCrop tem dados válidos
        if imgCrop.size != 0:
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

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Mostrar o frame e o canvas
    cv2.imshow("Output", img)
    cv2.imshow("Paint", paintWindow)

    # Tecla para salvar a imagem ou encerrar
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Imagem {counter} salva!")
    elif key == ord('d'):  # Limpar tela ao pressionar a tecla "d"
        points = [deque(maxlen=512)]
        index = 0
        paintWindow[67:, :, :] = 255  # Limpar canvas
    elif key == ord('q'):
        break

# Liberar a webcam e fechar janelas
cap.release()
cv2.destroyAllWindows()
