import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def print_out(self):
    headlen = len(f'Vertex cache of size: {len(self.cache)}:')
    print('=' * headlen)
    print(f'Vertex cache of size: {len(self.cache)}:')
    print('=' * headlen)
    for v in self.cache:
        self.cache[v].print_out()