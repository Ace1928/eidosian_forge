from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def untried_match(self):
    """
        :return: Whether the first element of the frontier is a token
            that has not yet been matched.
        :rtype: bool
        """
    if len(self._rtext) == 0:
        return False
    tried_matches = self._tried_m.get(self._freeze(self._tree), [])
    return self._rtext[0] not in tried_matches