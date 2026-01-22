import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def dict_suggest(dict, word):
    num_suggs_p = pointer(c_size_t(0))
    suggs_c = dict_suggest1(dict, word, len(word), num_suggs_p)
    suggs = []
    n = 0
    while n < num_suggs_p.contents.value:
        suggs.append(suggs_c[n])
        n = n + 1
    if num_suggs_p.contents.value > 0:
        dict_free_string_list(dict, suggs_c)
    return suggs