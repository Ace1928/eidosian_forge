import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def merge_as_sets(*args):
    """Create an order set (represented as a list) of the objects in
    the sequences passed as arguments."""
    s = {}
    for x in args:
        for y in x:
            s[y] = True
    return sorted(s)