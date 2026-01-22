import functools
import re
import nltk.tree
def treepositions_no_leaves(tree):
    """
    Returns all the tree positions in the given tree which are not
    leaf nodes.
    """
    treepositions = tree.treepositions()
    prefixes = set()
    for pos in treepositions:
        for length in range(len(pos)):
            prefixes.add(pos[:length])
    return [pos for pos in treepositions if pos in prefixes]