import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
def load_sparse_tensor(filename):
    loader = torch.load(filename)
    matrix = torch.sparse.FloatTensor(loader['indices'], loader['values'], loader['size'])
    return (matrix, loader['metadata'] if 'metadata' in loader else None)