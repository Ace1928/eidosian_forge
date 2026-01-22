from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def order_profile(G, name=None):
    """Determine how many elements a group has, of each order. """
    elts = elts_by_order(G)
    profile = {o: len(e) for o, e in elts.items()}
    if name:
        print(f'{name}: ' + ' '.join((f'{len(profile[r])}@{r}' for r in sorted(profile.keys()))))
    return profile