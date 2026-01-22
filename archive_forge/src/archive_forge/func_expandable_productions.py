from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def expandable_productions(self):
    """
        :return: A list of all the productions for which expansions
            are available for the current parser state.
        :rtype: list(Production)
        """
    if len(self._frontier) == 0:
        return []
    frontier_child = self._tree[self._frontier[0]]
    if len(self._frontier) == 0 or not isinstance(frontier_child, Tree):
        return []
    return [p for p in self._grammar.productions() if p.lhs().symbol() == frontier_child.label()]