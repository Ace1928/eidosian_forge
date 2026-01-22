import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def treeposition(self):
    """
        The tree position of this tree, relative to the root of the
        tree.  I.e., ``ptree.root[ptree.treeposition] is ptree``.
        """
    if self.parent() is None:
        return ()
    else:
        return self.parent().treeposition() + (self.parent_index(),)