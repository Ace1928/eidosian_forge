from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def collapsed_tree(self, *path_to_tree):
    """
        Return the ``TreeSegmentWidget`` for the specified subtree.

        :param path_to_tree: A list of indices i1, i2, ..., in, where
            the desired widget is the widget corresponding to
            ``tree.children()[i1].children()[i2]....children()[in]``.
            For the root, the path is ``()``.
        """
    return self._collapsed_trees[path_to_tree]