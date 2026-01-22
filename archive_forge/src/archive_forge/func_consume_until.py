import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def consume_until(self, c):
    if callable(c):
        i = 0
        while i < len(self.s) and (not c(self.s[i])):
            i = i + 1
        return self.advance(i)
    else:
        i = self.s.index(c)
        res = self.advance(i)
        self.advance(len(c))
        return res