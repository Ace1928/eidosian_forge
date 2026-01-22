from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def kmp_table(word):
    """Build the 'partial match' table of the Knuth-Morris-Pratt algorithm.

    Note: This is applicable to strings or
    quantum circuits represented as tuples.
    """
    pos = 2
    cnd = 0
    table = []
    table.append(-1)
    table.append(0)
    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1
    return table