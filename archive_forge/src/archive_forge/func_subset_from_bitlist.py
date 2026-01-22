from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@classmethod
def subset_from_bitlist(self, super_set, bitlist):
    """
        Gets the subset defined by the bitlist.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset
        ['c', 'd']

        See Also
        ========

        bitlist_from_subset
        """
    if len(super_set) != len(bitlist):
        raise ValueError('The sizes of the lists are not equal')
    ret_set = []
    for i in range(len(bitlist)):
        if bitlist[i] == '1':
            ret_set.append(super_set[i])
    return Subset(ret_set, super_set)