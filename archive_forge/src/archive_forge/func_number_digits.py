import re
import numpy as np
def number_digits(n):
    return int(np.floor(np.log10(np.abs(n))) + 1)