import numpy
import cv2
import os

def get_data():
    filepath = "specgram_my/" #添加路径
    filename= os.listdir(filepath)
    images = list()
    label = list()
    for it in filename:
        #print(it)
        category = it.split("-")[1]
        img = cv2.imread(filepath+it,0)
        img = cv2.resize(img,(64,64))
        images.append(img)
        label.append(category)

    X = numpy.array(images)
    y = numpy.array(label)
    print(X.shape, y.shape)
    return X, y


def get_test_data():
    X, y = get_data()
    return X[:11194], X[-1200:], y[:11194], y[-1200:]
     #return train_test_split(X, y, test_size=0.3, random_state=30)
    #a = numpy.load("a.npy")
    #b = numpy.load("b.npy")
    #c = numpy.load("c.npy")
    #d = numpy.load("d.npy")
    #return a,b,c,d


if __name__ == "__main__":
    a, b, c, d = get_test_data()
    numpy.save("a.npy",a)
    numpy.save("b.npy",b)
    numpy.save("c.npy",c)
    numpy.save("d.npy",d)
    print(c.shape, d.shape)