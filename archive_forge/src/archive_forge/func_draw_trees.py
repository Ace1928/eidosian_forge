from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def draw_trees(*trees):
    """
    Open a new window containing a graphical diagram of the given
    trees.

    :rtype: None
    """
    TreeView(*trees).mainloop()
    return