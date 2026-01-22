import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def genColors(n=100):
    out = [None] * n
    for i in range(n):
        x = float(i) / n
        out[i] = Color(redfunc(x), greenfunc(x), bluefunc(x))
    return out