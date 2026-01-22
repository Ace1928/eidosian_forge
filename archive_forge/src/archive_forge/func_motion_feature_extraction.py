from os.path import dirname
from os.path import join
import numpy as np
import scipy.fftpack
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.stats
from ..motion import blockMotion
from ..utils import *
def motion_feature_extraction(frames):
    frames = frames.astype(np.float32)
    mblock = 10
    h = gen_gauss_window(2, 0.5)
    motion_vectors = blockMotion(frames, method='N3SS', mbSize=mblock, p=np.int(1.5 * mblock))
    motion_vectors = motion_vectors.astype(np.float32)
    Eigens = np.zeros((motion_vectors.shape[0], motion_vectors.shape[1], motion_vectors.shape[2], 2), dtype=np.float32)
    for i in range(motion_vectors.shape[0]):
        motion_frame = motion_vectors[i]
        upper_left = np.zeros_like(motion_frame[:, :, 0])
        lower_right = np.zeros_like(motion_frame[:, :, 0])
        off_diag = np.zeros_like(motion_frame[:, :, 0])
        scipy.ndimage.correlate1d(motion_frame[:, :, 0] ** 2, h, 0, upper_left, mode='reflect')
        scipy.ndimage.correlate1d(upper_left, h, 1, upper_left, mode='reflect')
        scipy.ndimage.correlate1d(motion_frame[:, :, 1] ** 2, h, 0, lower_right, mode='reflect')
        scipy.ndimage.correlate1d(lower_right, h, 1, lower_right, mode='reflect')
        scipy.ndimage.correlate1d(motion_frame[:, :, 1] * motion_frame[:, :, 0], h, 0, off_diag, mode='reflect')
        scipy.ndimage.correlate1d(off_diag, h, 1, off_diag, mode='reflect')
        for y in range(motion_vectors.shape[1]):
            for x in range(motion_vectors.shape[2]):
                mat = np.array([[upper_left[y, x], off_diag[y, x]], [off_diag[y, x], lower_right[y, x]]])
                w, _ = np.linalg.eig(mat)
                Eigens[i, y, x] = w
    num = (Eigens[:, :, :, 0] - Eigens[:, :, :, 1]) ** 2
    den = (Eigens[:, :, :, 0] + Eigens[:, :, :, 1]) ** 2
    Coh10x10 = np.zeros_like(num)
    Coh10x10[den != 0] = num[den != 0] / den[den != 0]
    meanCoh10x10 = np.mean(Coh10x10)
    mode10x10 = np.zeros(motion_vectors.shape[0], dtype=np.float32)
    mean10x10 = np.zeros(motion_vectors.shape[0], dtype=np.float32)
    for i in range(motion_vectors.shape[0]):
        motion_frame = motion_vectors[i]
        motion_amplitude = np.sqrt(motion_vectors[i, :, :, 0] ** 2 + motion_vectors[i, :, :, 1] ** 2)
        mode10x10[i] = scipy.stats.mode(motion_amplitude, axis=None)[0]
        mean10x10[i] = np.mean(motion_amplitude)
    motion_diff = np.abs(mode10x10 - mean10x10)
    G = np.mean(motion_diff) / (1 + np.mean(mode10x10))
    return np.array([meanCoh10x10, G])