import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Carrega os dados
data_dict = pickle.load(open('./data_movimento_sequencia.pickle', 'rb'))

# Converte os dados e labels para arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Certifica-se de que as labels estejam em um formato categórico
labels = to_categorical(labels)

# Divide os dados em treino e teste (usando as sequências)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Definindo o modelo LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))  # 64 unidades LSTM
model.add(Dense(y_train.shape[1], activation='softmax'))  # Saída correspondente às classes

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo com as sequências
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Faz previsões no conjunto de teste
y_predict = model.predict(x_test)

# Converte as previsões e o conjunto de teste de volta às classes (não categóricas)
y_test_classes = np.argmax(y_test, axis=1)
y_predict_classes = np.argmax(y_predict, axis=1)

# Calcula a acurácia
score = accuracy_score(y_predict_classes, y_test_classes)
print('{}% das amostras foram classificadas corretamente!'.format(score * 100))

# Salva o modelo treinado
model.save('model_sequencia.h5')
