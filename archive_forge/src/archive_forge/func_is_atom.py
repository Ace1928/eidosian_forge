from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
@staticmethod
def is_atom(e):
    if isinstance(e, NegatedExpression):
        e = e.term
    if isinstance(e, ApplicationExpression):
        for arg in e.args:
            if not TableauProver.is_atom(arg):
                return False
        return True
    elif isinstance(e, AbstractVariableExpression) or isinstance(e, LambdaExpression):
        return True
    else:
        return False