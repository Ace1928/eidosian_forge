import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def patch_lines(lines, patches):
    """Applies patches to lines.  Updates lines in place."""
    for first, last, args in patches:
        lines[first:last] = args