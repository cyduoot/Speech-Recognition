import os
import wave

import cv2
import numpy as np
from PIL import Image
from keras import models
from spec import getFrequencyMagnitudes
from record import record




def wavread(path):# 读取wav文件，返回一个声道的数据和帧率
    wavfile = wave.open(path, "rb")
    params = wavfile.getparams()
    framesrate, frameswav = params[2], params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav, dtype=np.short)  # 读取wav数据
    return datause, framesrate

def spetrogram(path):
    data, framesrate = wavread(path)
    window_l = int(framesrate / 80)
    step_l = int(window_l / 2) # 计算相应的窗长和步长

    # 对每个窗计算频率强度
    freq_mag_list = []
    length = len(data)
    for i in range(0, length - window_l, step_l):
        frame = data[i:i + window_l]
        windowed_frame = frame * np.hamming(window_l)
        freq_mag_list.append(getFrequencyMagnitudes(np.fft.fft(windowed_frame))) # 这里为了效率，虽然自己写了FFT，但是和numpy提供的速度差距太大了，就仍使用numpy的fft

    # 寻找最大值以调整规模
    max_freq = float('-inf')
    for row in freq_mag_list:
        max_freq = max(max_freq, max(row))

    # 调整好数据规模，映射到0-255并且进行相应的旋转
    image_pixel_lst = []
    for row in freq_mag_list:
        pixel_row = []
        for freq in row[:int(len(row) / 2)]:
            pixel_row.append([255 * (1 - (freq / max_freq))] * 3)
        image_pixel_lst.append(pixel_row)

    image_pixel_lst = np.array(image_pixel_lst).astype('uint8')
    image_pixel_lst = np.rot90(image_pixel_lst, k = 1)

    visual = Image.fromarray(image_pixel_lst)
    return visual


def predict(model):
    #model = models.load_model("resnet.h5")

    record()
    img = spetrogram("output.wav")
    img.save("output.png")
    img = cv2.imread("output.png", 0)
    img = cv2.resize(img, (224, 224))

    X = img.reshape(1, 224, 224, 1)
    X = X.astype('float32')
    X /= 255
    label = ['语音','余音','识别','失败','中国','忠告','北京','背景','上海','商行','复旦','饭店',
     'speech','speaker','signal','file','print','open','close','project']
    result = model.predict(X).tolist()[0]
    ret = []
    for i in range(3):
        t = result.index(max(result))
        ret.append((label[t], result[t]))
        result[t]= - 1



if __name__ == "__main__":
    model = models.load_model("resnet.h5")
    predict(model)