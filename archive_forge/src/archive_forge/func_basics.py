import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def basics(canvasClass):
    """A general test of most of the drawing primitives except images and strings."""
    canvas = canvasClass((300, 300), 'test-basics')
    return drawBasics(canvas)