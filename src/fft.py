import cmath
from math import sin, cos, pi, log

# 复数单位i
i = complex(0.0, 1.0)

def fft(x):
    x = zeropadding(x)
    w = -i * 2.0 * pi / n
    n = len(x)


    if n == 1:
        return [complex(x[0])]
    else:
        e = []
        o = []  #分别是偶数位置和奇数位置，各自进行fft
        for j in range(0, n, 2):
            e.append(x[j])
            o.append(x[j+1])
        e = fft(e)
        o = fft(o)
        x = [0]*n
        for k in range(n//2):  # 各自fft结束之后，根据碟形计算方案计算出x的每个位置的fft值
            t = cmath.exp(w * k) * o[k]
            x[k] = e[k] + t
            x[k + (n//2)] = e[k] - t
        return x

def zeropadding(x):  # 对不足2的幂次的进行补零
    n = len(x)
    if log(n, 2) == int(log(n, 2)):
        return x
    else:
        n = round(log(n, 2) + .5)
        k = (2 ** n) - n
        x = x + ([0] * k)
        return x