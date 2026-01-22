import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def rotstring(canvasClass):
    canvas = canvasClass((450, 300), name='test-rotstring')
    return drawRotstring(canvas)