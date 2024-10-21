import os
import pickle
import cv2
import numpy as np
import mediapipe as mp

# Inicializa o modelo de detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Define o diretório de dados
DATA_DIR = './data_moviment'

# Parâmetro para definir o número de frames em uma sequência
SEQUENCIA_TAMANHO = 20

# Inicializa listas para armazenar os dados e labels
data_movimento = []
labels_movimento = []

# Função para calcular o optical flow entre dois frames
def calcular_optical_flow(prev_frame, current_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

# Função para normalizar uma sequência de movimentos entre 0 e 1
def normalizar_sequencia(sequencia):
    sequencia = np.array(sequencia)
    min_valor = np.min(sequencia)
    max_valor = np.max(sequencia)
    return (sequencia - min_valor) / (max_valor - min_valor + 1e-5)

# Função para criar uma máscara de pele
def criar_mascara_pele(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Aplicar uma operação de dilatação para melhorar a máscara
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    return skin_mask

# Processa cada vídeo no diretório de dados
for dir_ in os.listdir(DATA_DIR):
    video_path = os.path.join(DATA_DIR, dir_)
    cap = cv2.VideoCapture(video_path)
    
    sequencia_movimento_atual = []
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Criar máscara de pele
        skin_mask = criar_mascara_pele(frame)

        # Aplicar a máscara ao frame original
        frame_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

        # Detecção de mãos
        frame_rgb = cv2.cvtColor(frame_skin, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Verifica se uma mão foi detectada
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Pega a primeira mão detectada

            # Aqui você pode calcular a posição da mão
            hand_position = np.array([
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            ])

            if prev_frame is not None:
                # Calcula o optical flow entre o frame anterior e o atual
                movimento_frame = calcular_optical_flow(prev_frame, frame_skin)  # Use frame_skin
                movimento_total = np.mean(movimento_frame)  # Movimento médio no frame
                
                # Adiciona o movimento total à sequência atual
                sequencia_movimento_atual.append(movimento_total)

                # Quando a sequência atingir o tamanho definido, normalizar e armazenar
                if len(sequencia_movimento_atual) == SEQUENCIA_TAMANHO:
                    sequencia_normalizada = normalizar_sequencia(sequencia_movimento_atual)
                    data_movimento.append(sequencia_normalizada)
                    labels_movimento.append(dir_)
                    sequencia_movimento_atual = []  # Reseta a sequência
        
        # Atualiza o frame anterior
        prev_frame = frame_skin  # Use frame_skin como o frame anterior

    cap.release()

# Salva os dados de movimento e labels em um arquivo pickle para usar no modelo LSTM
with open('data_movimento_opticalflow.pickle', 'wb') as f:
    pickle.dump({'data': data_movimento, 'labels': labels_movimento}, f)

print('Processo de coleta finalizado.')
cv2.destroyAllWindows()
