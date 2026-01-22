from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def untried_expandable_productions(self):
    """
        :return: A list of all the untried productions for which
            expansions are available for the current parser state.
        :rtype: list(Production)
        """
    tried_expansions = self._tried_e.get(self._freeze(self._tree), [])
    return [p for p in self.expandable_productions() if p not in tried_expansions]