from ..utils import *
import numpy as np
import scipy.ndimage
import scipy.linalg
def est_params(frame, blk, sigma_nn):
    h, w = frame.shape
    sizeim = np.floor(np.array(frame.shape) / blk) * blk
    sizeim = sizeim.astype(np.int)
    frame = frame[:sizeim[0], :sizeim[1]]
    temp = []
    for u in range(blk):
        for v in range(blk):
            temp.append(np.ravel(frame[v:sizeim[0] - (blk - v) + 1, u:sizeim[1] - (blk - u) + 1]))
    temp = np.array(temp).astype(np.float32)
    cov_mat = np.cov(temp, bias=1).astype(np.float32)
    eigval, eigvec = np.linalg.eig(cov_mat)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    cov_mat = Q * xdiag * Q.T
    temp = []
    for u in range(blk):
        for v in range(blk):
            temp.append(np.ravel(frame[v::blk, u::blk]))
    temp = np.array(temp).astype(np.float32)
    V, d = scipy.linalg.eigh(cov_mat.astype(np.float64))
    V = V.astype(np.float32)
    sizeim_reduced = (sizeim / blk).astype(np.int)
    ss = np.zeros((sizeim_reduced[0], sizeim_reduced[1]), dtype=np.float32)
    if np.max(V) > 0:
        ss = scipy.linalg.solve(cov_mat, temp)
        ss = np.sum(np.multiply(ss, temp) / blk ** 2, axis=0)
        ss = ss.reshape(sizeim_reduced)
    V = V[V > 0]
    ent = np.zeros_like(ss, dtype=np.float32)
    for u in range(V.shape[0]):
        ent += np.log2(ss * V[u] + sigma_nn) + np.log(2 * np.pi * np.exp(1))
    return (ss, ent)