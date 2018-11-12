import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont
from predict import spetrogram, wavread
from keras import models
import pyaudio
import wave
import cv2


class PushButton(QWidget):
    def __init__(self):
        super(PushButton, self).__init__()
        self.initUI()

    def record(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 48000
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = "output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        self.statusLabel.setText("正在录音")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")
        self.statusLabel.setText("录音结束")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def load(self):
        self.model = models.load_model("resnet.h5")
        self.statusLabel.setText("模型加载完毕")
        self.statusLabel.setScaledContents(True)

    def predict(self):
        img = spetrogram("output.wav")
        img.save("output.png")
        img = cv2.imread("output.png", 0)
        img = cv2.resize(img, (224, 224))

        X = img.reshape(1, 224, 224, 1)
        X = X.astype('float32')
        X /= 255
        label = ['语音','余音','识别','失败','中国','忠告','北京','背景','上海','商行','复旦','饭店',
                 'speech','speaker','signal','file','print','open','close','project']
        result = self.model.predict(X).tolist()[0]
        ret = []
        for i in range(3):
            t = result.index(max(result))
            ret.append((label[t], result[t]))
            result[t]= - 1
        result = ret
        ret = list()
        for i in result:
            ret.append(i[0] + " " + str(round(i[1], 2)))
        self.firstresult.setText(ret[0])
        self.secondresult.setText(ret[1])

    def initUI(self):
        self.setWindowTitle("语音识别")
        self.setGeometry(400, 400, 300, 160)

        self.loadButton = QPushButton(self)
        self.loadButton.setText("Load")  # text
        self.loadButton.clicked.connect(self.load)
        self.loadButton.setToolTip("Load the model")  # Tool tip
        self.loadButton.move(20, 60)

        self.recordButton = QPushButton(self)
        self.recordButton.setText("Record")
        self.recordButton.clicked.connect(self.record)
        self.recordButton.setToolTip("Record your voice, 2 second")
        self.recordButton.move(120, 60)

        self.predictButton = QPushButton(self)
        self.predictButton.setText("Predict")  # text
        self.predictButton.clicked.connect(self.predict)
        self.predictButton.setToolTip("Predict the voice just recorded")  # Tool tip
        self.predictButton.move(220, 60)

        self.statusLabel = QLabel(self)
        font = QFont()
        font.setPointSize(15)
        self.statusLabel.setFont(font)
        self.statusLabel.setText("模型未加载")
        self.statusLabel.setScaledContents(True)
        self.statusLabel.setFixedWidth(400)
        self.statusLabel.move(0, 100)

        self.firstresult = QLabel(self)
        font = QFont()
        font.setPointSize(15)
        self.firstresult.setFont(font)
        self.firstresult.setText("概率最高结果")
        self.firstresult.setScaledContents(True)
        self.firstresult.setFixedWidth(400)
        self.firstresult.move(0, 0)

        self.secondresult = QLabel(self)
        font = QFont()
        font.setPointSize(15)
        self.secondresult.setFont(font)
        self.secondresult.setText("概率次高结果")
        self.secondresult.setScaledContents(True)
        self.secondresult.setFixedWidth(400)
        self.secondresult.move(0, 30)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PushButton()
    ex.show()
    sys.exit(app.exec_())
