import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def greenfunc(x):
    return 1 - pow(redfunc(x + 0.2), 2) - bluefunc(x - 0.3)