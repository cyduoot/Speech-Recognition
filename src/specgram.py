import wave
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from spec import spetrogram
filepath = filepath = "data/" #添加路径
filename= os.listdir(filepath)
for it in filename:

    print(it)
    name = it.split(".")
    if os.path.exists("specgram_my/"+name[0]+".png"):
        continue
    spec = spetrogram(filepath + it)
    image = spec.resize((224, 224),Image.ANTIALIAS)
    image = image.convert('L')
    image.save("specgram_my/"+name[0]+".png", format='png')
    image.close()

#生成matplotlib库提供的语谱图
'''
filepath = filepath = "data/" #添加路径
filename= os.listdir(filepath)
for it in filename:
    print(it)
    f = wave.open(filepath + it,'rb')
    name = it.split(".")
    #if os.path.exists("specgram/"+name[0]+".png"):
    #    continue
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes - 1,nchannels]).T
    f.close()
    # plot the wave
    plt.specgram(waveData[0],Fs = framerate, scale_by_freq = True, sides = 'default')
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(s)')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(0.64/3,0.64/3)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig("specgram_small/"+name[0]+".png", format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.close()

'''
