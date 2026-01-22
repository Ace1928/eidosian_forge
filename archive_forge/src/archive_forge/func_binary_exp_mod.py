from __future__ import absolute_import, division, print_function
import sys
def binary_exp_mod(f, e, m):
    """Computes f^e mod m in O(log e) multiplications modulo m."""
    len_e = -1
    x = e
    while x > 0:
        x >>= 1
        len_e += 1
    result = 1
    for k in range(len_e, -1, -1):
        result = result * result % m
        if e >> k & 1 != 0:
            result = result * f % m
    return result