from itertools import chain
def inputAlphabet(self):
    """
        The full set of symbols acceptable to this automaton.
        """
    return {inputSymbol for inState, inputSymbol, outState, outputSymbol in self._transitions}