import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def productions(self):
    """
        Generate the productions that correspond to the non-terminal nodes of the tree.
        For each subtree of the form (P: C1 C2 ... Cn) this produces a production of the
        form P -> C1 C2 ... Cn.

            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.productions() # doctest: +NORMALIZE_WHITESPACE
            [S -> NP VP, NP -> D N, D -> 'the', N -> 'dog', VP -> V NP, V -> 'chased',
            NP -> D N, D -> 'the', N -> 'cat']

        :rtype: list(Production)
        """
    if not isinstance(self._label, str):
        raise TypeError('Productions can only be generated from trees having node labels that are strings')
    prods = [Production(Nonterminal(self._label), _child_names(self))]
    for child in self:
        if isinstance(child, Tree):
            prods += child.productions()
    return prods