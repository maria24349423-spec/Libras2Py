import os
import cv2

# Define o diretório onde os vídeos serão salvos
DATA_DIR = './data_moviment'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 6  # Número de classes
dataset_size = 10  # Número de vídeos a serem gravados

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    # Cria diretório para a classe se não existir
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Coletando dados para a classe {}'.format(j))

    done = False

    # Espera o usuário pressionar 'Q' para começar a gravação
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pronto? Aperte "Q" para começar!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == ord('q'):
            done = True

    # Gravação dos vídeos
    for counter in range(dataset_size):
        # Define o nome do arquivo de saída
        video_path = os.path.join(class_dir, '{}.avi'.format(counter))
        
        # Define o codec e cria o objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        print('Gravando vídeo {} de {}'.format(counter + 1, dataset_size))

        # Grava por 5 segundos
        start_time = cv2.getTickCount()
        duration = 5  # Duração em segundos
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            remaining_time = duration - elapsed_time
            
            # Adiciona o contador no frame
            cv2.putText(frame, 'Gravando... {}s restantes'.format(int(remaining_time)), 
                        (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            out.write(frame)  # Grava o frame no arquivo
            cv2.imshow('Gravando', frame)

            # Para a gravação após 5 segundos
            if remaining_time <= 0:  # 5 segundos
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite parar a gravação com 'Q'
                break

        out.release()  # Libera o objeto VideoWriter
        print('Vídeo {} salvo.'.format(counter))

cap.release()  # Libera a captura da câmera
cv2.destroyAllWindows()  # Fecha todas as janelas
print('Processo de coleta finalizado.')

# Função para exibir o vídeo processado com Optical Flow
def exibir_video_optical_flow(video_path):
    cap_optical = cv2.VideoCapture(video_path)

    while cap_optical.isOpened():
        ret, frame = cap_optical.read()
        if not ret:
            break
        
        # Aqui você pode adicionar o processamento de Optical Flow ao frame se desejar
        # Por exemplo, exibir a imagem original
        cv2.imshow('Vídeo Processado com Optical Flow', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Permite fechar o vídeo com 'Q'
            break

    cap_optical.release()
    cv2.destroyAllWindows()

# Caminho do vídeo processado que você deseja abrir
# Altere o caminho para o vídeo que você já processou
video_processado_path = './caminho/do/video_processado.avi'  # Substitua pelo caminho correto
exibir_video_optical_flow(video_processado_path)