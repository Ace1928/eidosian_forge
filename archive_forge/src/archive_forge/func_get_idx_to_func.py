import urllib.request
import sys
from typing import Tuple
def get_idx_to_func(cu_h, cu_func):
    cu_sig = cu_h.find(cu_func)
    while True:
        if cu_sig == -1:
            break
        elif cu_h[cu_sig + len(cu_func)] != '(':
            cu_sig = cu_h.find(cu_func, cu_sig + 1)
        else:
            break
    return cu_sig