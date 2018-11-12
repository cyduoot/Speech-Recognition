import wave
from typing import List, Any, Union

import numpy as np
import math
from PIL import Image
import sys


def getFrequencyMagnitudes(trans_list):
    """
    根据输入的傅里叶变换值计算频率强度
    """
    result = []

    for i in trans_list:
        # 取实部和虚部的平方和
        square_mag = (i.real ** 2 + i.imag ** 2) ** (0.5)

        # 取对数
        try:
            log_sq_mag = 10 * math.log10(square_mag)
        except:
            log_sq_mag = 0

        result.append(log_sq_mag)

    return result

def wavread(path):# 读取wav文件，返回一个声道的数据和帧率
    wavfile = wave.open(path, "rb")
    params = wavfile.getparams()
    framesrate, frameswav = params[2], params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav, dtype=np.short)  # 读取wav数据
    datause.shape = -1, 2
    datause = datause.T  # 分离数据的奇偶项，返回只需采用一个声道即可
    return datause[0], framesrate

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


if __name__ == "__main__":
    spetrogram("output.wav").save("sample.png")