import asyncore
from collections import deque
from warnings import _deprecated
def find_prefix_at_end(haystack, needle):
    l = len(needle) - 1
    while l and (not haystack.endswith(needle[:l])):
        l -= 1
    return l