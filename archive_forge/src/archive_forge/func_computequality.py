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
def computequality(img, blocksizerow, blocksizecol, mu_prisparam, cov_prisparam):
    img = img[:, :, 0]
    h, w = img.shape
    if h < blocksizerow or w < blocksizecol:
        print('Input frame is too small')
        exit(0)
    hoffset = h % blocksizerow
    woffset = w % blocksizecol
    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]
    img = img.astype(np.float32)
    img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
    mscn1, var, mu = compute_image_mscn_transform(img, extend_mode='nearest')
    mscn1 = mscn1.astype(np.float32)
    mscn2, _, _ = compute_image_mscn_transform(img2, extend_mode='nearest')
    mscn2 = mscn2.astype(np.float32)
    feats_lvl1 = extract_on_patches(mscn1, blocksizerow, blocksizecol)
    feats_lvl2 = extract_on_patches(mscn2, blocksizerow / 2, blocksizecol / 2)
    feats = np.hstack((feats_lvl1, feats_lvl2))
    mu_distparam = np.mean(feats, axis=0)
    cov_distparam = np.cov(feats.T)
    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam) / 2)
    xd = mu_prisparam - mu_distparam
    quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]
    return np.hstack((mu_distparam, [quality]))