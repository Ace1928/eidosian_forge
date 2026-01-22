from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def replace_quants(self, ex, domain):
    """
        Apply the closed domain assumption to the expression

        - Domain = union([e.free()|e.constants() for e in all_expressions])
        - translate "exists x.P" to "(z=d1 | z=d2 | ... ) & P.replace(x,z)" OR
                    "P.replace(x, d1) | P.replace(x, d2) | ..."
        - translate "all x.P" to "P.replace(x, d1) & P.replace(x, d2) & ..."

        :param ex: ``Expression``
        :param domain: set of {Variable}s
        :return: ``Expression``
        """
    if isinstance(ex, AllExpression):
        conjuncts = [ex.term.replace(ex.variable, VariableExpression(d)) for d in domain]
        conjuncts = [self.replace_quants(c, domain) for c in conjuncts]
        return reduce(lambda x, y: x & y, conjuncts)
    elif isinstance(ex, BooleanExpression):
        return ex.__class__(self.replace_quants(ex.first, domain), self.replace_quants(ex.second, domain))
    elif isinstance(ex, NegatedExpression):
        return -self.replace_quants(ex.term, domain)
    elif isinstance(ex, ExistsExpression):
        disjuncts = [ex.term.replace(ex.variable, VariableExpression(d)) for d in domain]
        disjuncts = [self.replace_quants(d, domain) for d in disjuncts]
        return reduce(lambda x, y: x | y, disjuncts)
    else:
        return ex