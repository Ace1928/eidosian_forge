from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def row_sparse2gensim(row_idx, csr_matrix):
    indices = csr_matrix.indices[csr_matrix.indptr[row_idx]:csr_matrix.indptr[row_idx + 1]]
    g_row = [(col_idx, csr_matrix[row_idx, col_idx]) for col_idx in indices]
    return g_row