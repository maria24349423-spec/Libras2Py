import cv2, time, threading
import numpy as np
from collections import deque
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3
import tkinter as tk

# ===== tela de saída (segunda tela) =====
class Display:
    def __init__(self, title="Tradução LIBRAS"):
        self.text = ""
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root, text="", font=("Arial", 64), wraplength=1400, justify="center")
        self.label.pack(expand=True, fill="both", padx=20, pady=20)
        self.root.after(80, self._tick)

    def _tick(self):
        self.label.config(text=self.text)
        self.root.after(80, self._tick)

    def start(self):
        threading.Thread(target=self.root.mainloop, daemon=True).start()

# ===== fala (opcional) em thread separada =====
engine = pyttsx3.init()
def speak(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

# ===== visão computacional =====
cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)  # HD
detector = HandDetector(maxHands=1, detectionCon=0.7)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = [l.strip() for l in open("Model/labels.txt", "r", encoding="utf-8").read().splitlines()]

history = deque(maxlen=15)   # suavização
last_spoken = ""
last_time = 0

# inicia a janela de exibição (arraste pro 2º monitor)
disp = Display()
disp.start()

while True:
    ok, img = cap.read()
    if not ok: break

    hands, img = detector.findHands(img)  # desenha mão por padrão
    if hands:
        x, y, w, h = hands[0]['bbox']
        offset = 25
        H, W = img.shape[:2]
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(W, x + w + offset), min(H, y + h + offset)
        crop = img[y1:y2, x1:x2]

        if crop.size != 0:
            white = np.ones((300, 300, 3), np.uint8) * 255
            aspect = (y2 - y1) / max(1, (x2 - x1))
            if aspect > 1:
                k = 300 / (y2 - y1); wCalc = int(k * (x2 - x1))
                resize = cv2.resize(crop, (wCalc, 300)); wGap = (300 - wCalc) // 2
                white[:, wGap:wGap+wCalc] = resize
            else:
                k = 300 / (x2 - x1); hCalc = int(k * (y2 - y1))
                resize = cv2.resize(crop, (300, hCalc)); hGap = (300 - hCalc) // 2
                white[hGap:hGap+hCalc, :] = resize

            _, idx = classifier.getPrediction(white, draw=False)
            history.append(idx)
            # voto por maioria nos últimos frames
            idx_mode = max(set(history), key=history.count)
            label = labels[idx_mode]
            cv2.putText(img, label, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

            # atualiza fala/tela com "debounce"
            if time.time() - last_time > 1.2 and label != last_spoken:
                disp.text = label               # aparece na segunda tela
                # speak(label)                  # descomente se quiser TTS
                last_spoken = label
                last_time = time.time()

    cv2.imshow("Libras2Py - Camera", img)
    if cv2.waitKey(1) & 0xFF == 27: break  # ESC para sair

cap.release(); cv2.destroyAllWindows()