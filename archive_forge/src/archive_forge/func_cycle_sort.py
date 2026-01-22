from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def cycle_sort(l):
    """
    Utility function which takes a list l and returns the minimum
    for the alphabetical order among all cyclic permutations of the list.
    """
    s = l
    for i in range(0, len(l)):
        temp = l[i:] + l[0:i]
        if temp < s:
            s = temp
    return s