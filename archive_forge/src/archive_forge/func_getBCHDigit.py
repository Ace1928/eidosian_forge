import re
import itertools
@staticmethod
def getBCHDigit(data):
    digit = 0
    while data != 0:
        digit += 1
        data >>= 1
    return digit