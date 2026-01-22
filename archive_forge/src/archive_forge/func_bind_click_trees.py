from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def bind_click_trees(self, callback, button=1):
    """
        Add a binding to all tree segments.
        """
    for tseg in list(self._expanded_trees.values()):
        tseg.bind_click(callback, button)
    for tseg in list(self._collapsed_trees.values()):
        tseg.bind_click(callback, button)