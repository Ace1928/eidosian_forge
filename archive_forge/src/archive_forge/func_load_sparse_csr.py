import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
def load_sparse_csr(filename):
    loader = np.load(filename + '.npz', allow_pickle=True)
    matrix = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return (matrix, loader['metadata'].item(0) if 'metadata' in loader else None)