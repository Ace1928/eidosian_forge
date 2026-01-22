import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def extract_C(C, i, j, m, n):
    row_ind = [k in i or k - m in j for k in range(m + n)]
    col_ind = [k in j or k - n in i for k in range(m + n)]
    return C[row_ind, :][:, col_ind]