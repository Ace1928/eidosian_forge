import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def get_last_line(self, suffix):
    line = 0
    if self._children_groups:
        children_group = self._children_groups[-1]
        last_leaf = _get_previous_leaf_if_indentation(children_group.last_line_offset_leaf)
        line = last_leaf.end_pos[0] + children_group.line_offset
        if _ends_with_newline(last_leaf, suffix):
            line -= 1
    line += len(split_lines(suffix)) - 1
    if suffix and (not suffix.endswith('\n')) and (not suffix.endswith('\r')):
        line += 1
    if self._node_children:
        return max(line, self._node_children[-1].get_last_line(suffix))
    return line