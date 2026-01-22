import math
def from_bin(self, b):
    if b == self._bins - 1:
        return float('inf')
    else:
        unscaled = b * (b + 1.0) / 2.0
        return unscaled * self._scale