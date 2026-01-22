import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
def treepos(self, tree):
    """
        Convert this pointer to a standard 'tree position' pointer,
        given that it points to the given tree.
        """
    if tree is None:
        raise ValueError('Parse tree not available')
    stack = [tree]
    treepos = []
    wordnum = 0
    while True:
        if isinstance(stack[-1], Tree):
            if len(treepos) < len(stack):
                treepos.append(0)
            else:
                treepos[-1] += 1
            if treepos[-1] < len(stack[-1]):
                stack.append(stack[-1][treepos[-1]])
            else:
                stack.pop()
                treepos.pop()
        elif wordnum == self.wordnum:
            return tuple(treepos[:len(treepos) - self.height - 1])
        else:
            wordnum += 1
            stack.pop()