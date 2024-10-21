import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Inicializa as soluções do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define diretório de dados
DATA_DIR = './data_moviment/A'

# Parâmetro para definir o número de frames em uma sequência
SEQUENCIA_TAMANHO = 49

# Inicializa listas para armazenar os dados e labels
data_movimento = []
data_estatico = []
labels_movimento = []
labels_estatico = []

# Função para calcular o movimento total da mão em um frame
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

# Função para normalizar uma sequência de movimentos entre 0 e 1
def normalizar_sequencia(sequencia):
    min_valor = min(sequencia)
    max_valor = max(sequencia)
    
    if max_valor - min_valor == 0:
        return [0] * len(sequencia) 
    
    return [(valor - min_valor) / (max_valor - min_valor) for valor in sequencia]

# Processa cada sequência de imagens no diretório de dados
for dir_ in os.listdir(DATA_DIR):
    sequencia_movimento_atual = []
    sequencia_estatica_atual = []
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                movimento_total = calcular_movimento_total(hand_landmarks, width, height)
                sequencia_movimento_atual.append(movimento_total)

                # Extrai as coordenadas estáticas dos landmarks
                coordenadas_estaticas = []
                for lm in hand_landmarks.landmark:
                    coordenadas_estaticas.append(lm.x)
                    coordenadas_estaticas.append(lm.y)
                
                # Adiciona as coordenadas estáticas ao frame
                sequencia_estatica_atual.append(coordenadas_estaticas)

                if len(sequencia_movimento_atual) == SEQUENCIA_TAMANHO:
                    sequencia_normalizada = normalizar_sequencia(sequencia_movimento_atual)
                    data_movimento.append(sequencia_normalizada)
                    labels_movimento.append(dir_)

                    data_estatico.append(sequencia_estatica_atual[-1])  # Última frame para sinais estáticos
                    labels_estatico.append(dir_)

                    sequencia_movimento_atual = []
                    sequencia_estatica_atual = []

# Salva os dados de movimento e estáticos em arquivos pickle
with open('data_movimento_sequencia_normalizada.pickle', 'wb') as f:
    pickle.dump({'data': data_movimento, 'labels': labels_movimento}, f)

with open('data_estatico.pickle', 'wb') as f:
    pickle.dump({'data': data_estatico, 'labels': labels_estatico}, f)

hands.close()
