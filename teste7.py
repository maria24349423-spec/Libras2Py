import os
import cv2
import numpy as np
import pickle

# Caminho dos vídeos gravados
DATA_DIR = './data_moviment'

data_movimento = []  # Para armazenar as sequências de patches de movimento da mão
labels_movimento = []  # Para armazenar os rótulos

# Função para extrair patches de uma região específica (mão)
def extrair_patches(frame, patch_size=(16, 16), stride=8):
    """
    Divide um frame em patches pequenos (blocos), com sobreposição especificada por stride.
    """
    h, w, c = frame.shape
    patches = []
    for i in range(0, h - patch_size[0] + 1, stride):
        for j in range(0, w - patch_size[1] + 1, stride):
            patch = frame[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)
    return np.array(patches)

# Função para detectar a mão no frame usando uma máscara de cor de pele
def detectar_mao(frame):
    # Converter o frame para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir limites para a detecção da cor da pele
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Criar uma máscara para a cor da pele
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar operações morfológicas para suavizar a máscara (dilatação e erosão)
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)

    # Encontrar contornos na máscara da pele
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Selecionar o maior contorno, presumidamente o da mão
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        # Extrair a região da mão
        hand_roi = frame[y:y+h, x:x+w]
        return hand_roi
    else:
        return None

# Função para processar o vídeo e gerar patches da mão
def processar_video_para_patches(video_path, patch_size=(16, 16), stride=8):
    cap = cv2.VideoCapture(video_path)
    frames_patches = []  # Lista para armazenar os patches do vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar a mão no frame
        mao = detectar_mao(frame)
        if mao is not None:
            # Redimensiona a região da mão para o tamanho desejado (por exemplo, 256x256)
            mao = cv2.resize(mao, (256, 256))

            # Extrai os patches da região da mão
            patches = extrair_patches(mao, patch_size, stride)
            frames_patches.append(patches)

    cap.release()
    return frames_patches

# Processa cada vídeo gravado
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            print(f'Processando vídeo: {video_path}')
            
            # Extrai os patches da região da mão no vídeo
            frames_patches = processar_video_para_patches(video_path)
            
            # Adiciona os patches e o rótulo à lista de dados
            data_movimento.append(frames_patches)
            labels_movimento.append(class_dir)  # Adiciona o rótulo da classe

# Salva os dados coletados em um arquivo pickle
data = {'data': data_movimento, 'labels': labels_movimento}
with open('data_movimento_transformer.pickle', 'wb') as f:
    pickle.dump(data, f)

print('Dados salvos em data_movimento_transformer.pickle')
