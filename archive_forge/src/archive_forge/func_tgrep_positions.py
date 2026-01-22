import functools
import re
import nltk.tree
def tgrep_positions(pattern, trees, search_leaves=True):
    """
    Return the tree positions in the trees which match the given pattern.

    :param pattern: a tgrep search pattern
    :type pattern: str or output of tgrep_compile()
    :param trees: a sequence of NLTK trees (usually ParentedTrees)
    :type trees: iter(ParentedTree) or iter(Tree)
    :param search_leaves: whether to return matching leaf nodes
    :type search_leaves: bool
    :rtype: iter(tree positions)
    """
    if isinstance(pattern, (bytes, str)):
        pattern = tgrep_compile(pattern)
    for tree in trees:
        try:
            if search_leaves:
                positions = tree.treepositions()
            else:
                positions = treepositions_no_leaves(tree)
            yield [position for position in positions if pattern(tree[position])]
        except AttributeError:
            yield []