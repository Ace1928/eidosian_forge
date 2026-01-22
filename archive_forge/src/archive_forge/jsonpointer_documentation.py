from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
Constructs a JsonPointer from a list of (unescaped) paths

        >>> JsonPointer.from_parts(['a', '~', '/', 0]).path == '/a/~0/~1/0'
        True
        