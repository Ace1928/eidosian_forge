from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def scan_check(self, alpha, word):
    """
        Another version of ``scan`` routine, described on, it checks whether
        `\\alpha` scans correctly under `word`, it is a straightforward
        modification of ``scan``. ``scan_check`` returns ``False`` (rather than
        calling ``coincidence``) if the scan completes incorrectly; otherwise
        it returns ``True``.

        See Also
        ========

        scan, scan_c, scan_and_fill, scan_and_fill_c

        """
    A_dict = self.A_dict
    A_dict_inv = self.A_dict_inv
    table = self.table
    f = alpha
    i = 0
    r = len(word)
    b = alpha
    j = r - 1
    while i <= j and table[f][A_dict[word[i]]] is not None:
        f = table[f][A_dict[word[i]]]
        i += 1
    if i > j:
        return f == b
    while j >= i and table[b][A_dict_inv[word[j]]] is not None:
        b = table[b][A_dict_inv[word[j]]]
        j -= 1
    if j < i:
        return False
    elif j == i:
        table[f][A_dict[word[i]]] = b
        table[b][A_dict_inv[word[i]]] = f
    return True