import re
import itertools
@staticmethod
def gexp(n):
    while n < 0:
        n += 255
    while n >= 256:
        n -= 255
    return EXP_TABLE[n]