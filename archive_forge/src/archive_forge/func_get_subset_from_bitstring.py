from sympy.core import Basic, Integer
import random
def get_subset_from_bitstring(super_set, bitstring):
    """
    Gets the subset defined by the bitstring.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import get_subset_from_bitstring
    >>> get_subset_from_bitstring(['a', 'b', 'c', 'd'], '0011')
    ['c', 'd']
    >>> get_subset_from_bitstring(['c', 'a', 'c', 'c'], '1100')
    ['c', 'a']

    See Also
    ========

    graycode_subsets
    """
    if len(super_set) != len(bitstring):
        raise ValueError('The sizes of the lists are not equal')
    return [super_set[i] for i, j in enumerate(bitstring) if bitstring[i] == '1']