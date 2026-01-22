import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
@classmethod
def fromlist(cls, l):
    """
        :type l: list
        :param l: a tree represented as nested lists

        :return: A tree corresponding to the list representation ``l``.
        :rtype: Tree

        Convert nested lists to a NLTK Tree
        """
    if type(l) == list and len(l) > 0:
        label = repr(l[0])
        if len(l) > 1:
            return Tree(label, [cls.fromlist(child) for child in l[1:]])
        else:
            return label