import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def most_general_unification(a, b, bindings=None):
    """
    Find the most general unification of the two given expressions

    :param a: ``Expression``
    :param b: ``Expression``
    :param bindings: ``BindingDict`` a starting set of bindings with which the
                     unification must be consistent
    :return: a list of bindings
    :raise BindingException: if the Expressions cannot be unified
    """
    if bindings is None:
        bindings = BindingDict()
    if a == b:
        return bindings
    elif isinstance(a, IndividualVariableExpression):
        return _mgu_var(a, b, bindings)
    elif isinstance(b, IndividualVariableExpression):
        return _mgu_var(b, a, bindings)
    elif isinstance(a, ApplicationExpression) and isinstance(b, ApplicationExpression):
        return most_general_unification(a.function, b.function, bindings) + most_general_unification(a.argument, b.argument, bindings)
    raise BindingException((a, b))