import numpy as np
import pytest
from opt_einsum import contract, contract_path
def build_views(string):
    chars = 'abcdefghij'
    sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4])
    sizes = {c: s for c, s in zip(chars, sizes)}
    views = []
    string = string.replace('...', 'ij')
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [sizes[x] for x in term]
        views.append(np.random.rand(*dims))
    return views