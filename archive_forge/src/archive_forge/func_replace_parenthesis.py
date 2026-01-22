import re
import warnings
from enum import Enum
from math import gcd
def replace_parenthesis(s):
    """Replace substrings of input that are enclosed in parenthesis.

    Return a new string and a mapping of replacements.
    """
    left, right = (None, None)
    mn_i = len(s)
    for left_, right_ in (('(/', '/)'), '()', '{}', '[]'):
        i = s.find(left_)
        if i == -1:
            continue
        if i < mn_i:
            mn_i = i
            left, right = (left_, right_)
    if left is None:
        return (s, {})
    i = mn_i
    j = s.find(right, i)
    while s.count(left, i + 1, j) != s.count(right, i + 1, j):
        j = s.find(right, j + 1)
        if j == -1:
            raise ValueError(f'Mismatch of {left + right} parenthesis in {s!r}')
    p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]
    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    v = s[i + len(left):j]
    r, d = replace_parenthesis(s[j + len(right):])
    d[k] = v
    return (s[:i] + k + r, d)