import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def prove(self, verbose=False):
    """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
    if self._result is None:
        self._result, clauses = self._prover._prove(self.goal(), self.assumptions(), verbose)
        self._clauses = clauses
        self._proof = ResolutionProverCommand._decorate_clauses(clauses)
    return self._result