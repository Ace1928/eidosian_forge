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
def temporal_dc_variation_feature_extraction(frames):
    frames = frames.astype(np.float32)
    mblock = 16
    mbsize = 16
    ih = np.int(frames.shape[1] / mbsize) * mbsize
    iw = np.int(frames.shape[2] / mbsize) * mbsize
    motion_vectors = blockMotion(frames, method='N3SS', mbSize=mblock, p=7)
    dct_motion_comp_diff = np.zeros((motion_vectors.shape[0], motion_vectors.shape[1], motion_vectors.shape[2]), dtype=np.float32)
    for i in range(motion_vectors.shape[0]):
        for y in range(motion_vectors.shape[1]):
            for x in range(motion_vectors.shape[2]):
                patchP = frames[i + 1, y * mblock:(y + 1) * mblock, x * mblock:(x + 1) * mblock, 0].astype(np.float32)
                patchI = frames[i, y * mblock + motion_vectors[i, y, x, 0]:(y + 1) * mblock + motion_vectors[i, y, x, 0], x * mblock + motion_vectors[i, y, x, 1]:(x + 1) * mblock + motion_vectors[i, y, x, 1], 0].astype(np.float32)
                diff = patchP - patchI
                t = scipy.fftpack.dct(scipy.fftpack.dct(diff, axis=1, norm='ortho'), axis=0, norm='ortho')
                dct_motion_comp_diff[i, y, x] = t[0, 0]
    dct_motion_comp_diff = dct_motion_comp_diff.reshape(motion_vectors.shape[0], -1)
    std_dc = np.std(dct_motion_comp_diff, axis=1)
    dt_dc_temp = np.abs(std_dc[1:] - std_dc[:-1])
    dt_dc_measure1 = np.mean(dt_dc_temp)
    return np.array([dt_dc_measure1])