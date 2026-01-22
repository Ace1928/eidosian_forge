import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def treeposition_spanning_leaves(self, start, end):
    """
        :return: The tree position of the lowest descendant of this
            tree that dominates ``self.leaves()[start:end]``.
        :raise ValueError: if ``end <= start``
        """
    if end <= start:
        raise ValueError('end must be greater than start')
    start_treepos = self.leaf_treeposition(start)
    end_treepos = self.leaf_treeposition(end - 1)
    for i in range(len(start_treepos)):
        if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
            return start_treepos[:i]
    return start_treepos