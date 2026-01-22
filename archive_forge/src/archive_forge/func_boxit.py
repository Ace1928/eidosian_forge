from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def boxit(canvas, text):
    big = ('helvetica', -16, 'bold')
    return BoxWidget(canvas, TextWidget(canvas, text, font=big), fill='green')