from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def color2(treeseg):
    treeseg.label()['fill'] = '#%06d' % random.randint(0, 9999)
    treeseg.label().child()['color'] = 'white'