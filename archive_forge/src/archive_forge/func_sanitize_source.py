from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
def sanitize_source(src):
    """Strip the 'coding' directive from python source code, if present.

  This is a workaround for https://bugs.python.org/issue18960. Also see PEP-0263.
  """
    src_lines = src.splitlines(True)
    for i, line in enumerate(src_lines[:2]):
        if _CODING_PATTERN.match(line):
            src_lines[i] = re.sub('#.*$', '# (removed coding)', line)
    return ''.join(src_lines)