import math
def to_bin(self, x):
    if x < 0.0:
        raise ValueError('Values less than 0.0 not accepted.')
    elif x > self._max:
        return self._bins - 1
    else:
        scaled = x / self._scale
        return int(-0.5 + math.sqrt(2.0 * scaled + 0.25))