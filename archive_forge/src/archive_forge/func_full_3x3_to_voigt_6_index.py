import numpy as np
def full_3x3_to_voigt_6_index(i, j):
    if i == j:
        return i
    return 6 - i - j