import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def version_compare(a, b):
    va = Version(a)
    vb = Version(b)
    if va < vb:
        return -1
    if va > vb:
        return 1
    return 0