import string
from ..sage_helper import _within_sage, sage_method
def reverse_word(word):
    L = list(word)
    L.reverse()
    return ''.join(L)