import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os

# Verifica se o arquivo existe e não está vazio
pickle_file_path = 'data_movimento_opticalflow.pickle'

if not os.path.exists(pickle_file_path) or os.path.getsize(pickle_file_path) == 0:
    raise FileNotFoundError(f"O arquivo {pickle_file_path} não existe ou está vazio.")

# Carregar os dados de movimento e labels
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

# Verifique se 'data' contém as chaves esperadas
if 'data' not in data or 'labels' not in data:
    raise KeyError("As chaves 'data' e 'labels' devem estar presentes no arquivo.")

# Padronizar as sequências de movimento (optical flow ou outras características)
max_sequence_length = max(len(seq) for seq in data['data'])
X = np.array([np.pad(seq, (0, max_sequence_length - len(seq)), mode='constant') for seq in data['data']])
y = np.array(data['labels'])  # Labels correspondentes

# Verifica se X ou y estão vazios
if X.size == 0:
    raise ValueError("O array de dados 'X' está vazio.")
if y.size == 0:
    raise ValueError("O array de labels 'y' está vazio.")

# Normalizar os dados X
X = X / np.max(X)  # Normalização para a faixa [0, 1]

# Verificar o formato de X e ajustar se necessário para (n_amostras, n_timesteps, 1)
if X.ndim == 2:  # Caso a estrutura de dados seja (n_amostras, SEQUENCIA_TAMANHO)
    X = X.reshape((X.shape[0], X.shape[1], 1))

# Transformar as labels em categorias numéricas (se forem strings)
unique_labels = np.unique(y)  # Obter as classes únicas
label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Mapear cada label para um número
y_numerico = np.array([label_map[label] for label in y])  # Converter as labels para numéricas

# Calcular o número de amostras por classe
n_classes = len(unique_labels)
n_samples = len(y_numerico)

# Garantir que o número de amostras no conjunto de teste seja suficiente para cobrir todas as classes
test_size = max(0.2, n_classes / n_samples)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_numerico, test_size=test_size, stratify=y_numerico)

# Converter as labels para one-hot encoding
y_train_categ = to_categorical(y_train, num_classes=n_classes)
y_test_categ = to_categorical(y_test, num_classes=n_classes)

# Criar o modelo GRU
model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Usando GRU
model.add(GRU(64, return_sequences=False))  # Segunda camada GRU
model.add(Dense(64, activation='relu'))  # Camada densa adicional
model.add(Dense(32, activation='relu'))  # Outra camada densa
model.add(Dense(n_classes, activation='softmax'))  # Camada de saída

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir callbacks para parar o treinamento cedo se não houver melhoria
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar o modelo
model.fit(X_train, y_train_categ, validation_data=(X_test, y_test_categ), epochs=100, batch_size=8, callbacks=[early_stopping])

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test_categ)
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')

# Salvar o modelo treinado
model.save('modelo_gru_aprimorado.keras')
