import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def substitute_bindings(self, bindings):
    """
        Replace every binding

        :param bindings: A list of tuples mapping Variable Expressions to the
            Expressions to which they are bound.
        :return: ``Clause``
        """
    return Clause([atom.substitute_bindings(bindings) for atom in self])