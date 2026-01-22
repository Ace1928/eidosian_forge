import skvideo.io
import skvideo.utils
import numpy as np
import os
import sys
def pattern_sinusoid(backend):
    sinusoid1d = np.zeros((100, 100))
    for i in range(100):
        sinusoid1d[i, :] = 127 * np.sin(2 * np.pi * i / 100) + 128
    skvideo.io.vwrite('sinusoid1d.yuv', sinusoid1d)
    videoData1 = skvideo.io.vread('sinusoid1d.yuv', width=100, height=100)
    skvideo.io.vwrite('sinusoid1d_resaved.yuv', videoData1)
    videoData2 = skvideo.io.vread('sinusoid1d_resaved.yuv', width=100, height=100)
    sinusoidDataOriginal = np.array(sinusoid1d[:, 1])
    sinusoidDataVideo1 = skvideo.utils.rgb2gray(videoData1[0])[0, :, 1, 0]
    sinusoidDataVideo2 = skvideo.utils.rgb2gray(videoData2[0])[0, :, 1, 0]
    floattopixel_mse = np.mean((sinusoidDataOriginal - sinusoidDataVideo1) ** 2)
    assert floattopixel_mse < 1, 'Possible conversion error between floating point and raw video. MSE=%f' % (floattopixel_mse,)
    pixeltopixel_mse = np.mean((sinusoidDataVideo1 - sinusoidDataVideo2) ** 2)
    assert pixeltopixel_mse == 0, 'Creeping error inside vread/vwrite.'
    os.remove('sinusoid1d.yuv')
    os.remove('sinusoid1d_resaved.yuv')