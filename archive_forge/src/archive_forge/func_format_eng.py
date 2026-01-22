import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
def format_eng(self, num):
    """
        Format a number in engineering notation, appending a letter
        representing the power of 1000 of the original number.
        Some examples:

        >>> format_eng(0)        # for self.places = 0
        '0'

        >>> format_eng(1000000)  # for self.places = 1
        '1.0 M'

        >>> format_eng(-1e-6)  # for self.places = 2
        '-1.00 µ'
        """
    sign = 1
    fmt = 'g' if self.places is None else f'.{self.places:d}f'
    if num < 0:
        sign = -1
        num = -num
    if num != 0:
        pow10 = int(math.floor(math.log10(num) / 3) * 3)
    else:
        pow10 = 0
        num = 0.0
    pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))
    mant = sign * num / 10.0 ** pow10
    if abs(float(format(mant, fmt))) >= 1000 and pow10 < max(self.ENG_PREFIXES):
        mant /= 1000
        pow10 += 3
    prefix = self.ENG_PREFIXES[int(pow10)]
    if self._usetex or self._useMathText:
        formatted = f'${mant:{fmt}}${self.sep}{prefix}'
    else:
        formatted = f'{mant:{fmt}}{self.sep}{prefix}'
    return formatted