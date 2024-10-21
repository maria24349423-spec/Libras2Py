import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

# Inicializar o mecanismo de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Função para falar em uma thread separada
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Carregar o modelo treinado no formato .keras
model = tf.keras.models.load_model('modelo_gru_aprimorado.keras')

# Dicionário de labels (ajuste conforme necessário)
labels_dict = {0: 'D', 1: 'C', 2: 'A', 3: 'F', 4: 'H', 5: 'L'}  # Ajuste o número de classes conforme o seu modelo

cap = cv2.VideoCapture(0)

# Definir parâmetros para calcular optical flow
prev_frame = None
optical_flow = cv2.calcOpticalFlowFarneback

# Parâmetros da sequência
SEQUENCIA_TAMANHO = 20  # Tamanho da sequência
sequencia = []

# Inicializar a variável para armazenar a predição
predicted_character = ""

# Variáveis de controle de tempo
prediction_time = 1
last_audio_time = time.time()
audio_interval = 2

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Converter o frame para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir limites para a detecção da cor da pele
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Criar uma máscara para a cor da pele
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar uma operação de dilatação e erosão para suavizar a máscara
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)

    # Encontrar contornos na máscara da pele
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se houver contornos, processe o maior
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Criar uma nova imagem para o Optical Flow
        hand_roi = skin_mask[y:y+h, x:x+w]

        # Redimensionar a região de interesse (ROI) para um tamanho fixo
        hand_roi = cv2.resize(hand_roi, (640, 480))

        # A imagem da mão já está em escala de cinza, portanto, não precisamos de outra conversão.
        if prev_frame is None:
            prev_frame = hand_roi
            continue

        # Certifique-se de que o prev_frame também esteja em escala de cinza e do mesmo tamanho
        prev_frame = cv2.resize(prev_frame, (640, 480))

        # Calcular o optical flow apenas na região da mão
        flow = optical_flow(prev_frame, hand_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Atualizar o frame anterior
        prev_frame = hand_roi

        # Calcular a magnitude e o ângulo do fluxo
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalizar a magnitude para um intervalo [0, 1]
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        # Calcular a magnitude média do fluxo e adicionar à sequência
        movimento_medio = np.mean(mag_norm)
        sequencia.append([movimento_medio])

        # Verificar o optical flow e a sequência
        print(f'Média do movimento capturado: {movimento_medio}')
        print(f'Sequência atual: {sequencia}')

        # Se a sequência atingir o tamanho definido, fazer a predição
        if len(sequencia) == SEQUENCIA_TAMANHO:
            # Converter a sequência em um array numpy e redimensionar para o formato (1, SEQUENCIA_TAMANHO, 1)
            sequencia_np = np.array(sequencia).reshape(1, SEQUENCIA_TAMANHO, 1)

            # Fazer a predição usando o modelo GRU
            prediction = model.predict(sequencia_np)
            predicted_class = np.argmax(prediction)

            # Debug: Verificar predições e probabilidades
            print(f"Predicted class: {predicted_class}, Predicted character: {labels_dict.get(predicted_class, 'Desconhecido')}")
            print(f"Prediction probabilities: {prediction}")

            # Converter a predição em um caractere usando o dicionário de labels
            predicted_character = labels_dict.get(predicted_class, "Desconhecido")

            # Verificar se o intervalo de tempo para o áudio foi atingido
            current_time = time.time()
            if current_time - last_audio_time >= audio_interval:
                last_audio_time = current_time
                threading.Thread(target=speak, args=(predicted_character,)).start()

            # Resetar a sequência após a predição
            sequencia = []

        # Exibir a predição na tela (somente se houver predição)
        if predicted_character:
            cv2.putText(frame, f'Predição: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar o fluxo óptico (magnitude normalizada)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow('Optical Flow', mag_norm)

    # Exibir a máscara da pele para depuração
    cv2.imshow('Skin Mask', skin_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
