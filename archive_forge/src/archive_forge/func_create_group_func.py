from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
def create_group_func(node):
    name = 'n%s' % counter[0]
    self._group_names_to_nodes[name] = node.varname
    counter[0] += 1
    return name