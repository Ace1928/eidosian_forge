import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def dictOf(key, value):
    """Helper to easily and clearly define a dictionary by specifying the respective patterns
       for the key and value.  Takes care of defining the C{L{Dict}}, C{L{ZeroOrMore}}, and C{L{Group}} tokens
       in the proper order.  The key pattern can include delimiting markers or punctuation,
       as long as they are suppressed, thereby leaving the significant key text.  The value
       pattern can include named results, so that the C{Dict} results can include named token
       fields.
    """
    return Dict(ZeroOrMore(Group(key + value)))