important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_name_of_position(self, position):
    """
        Given a (line, column) tuple, returns a :py:class:`Name` or ``None`` if
        there is no name at that position.
        """
    for c in self.children:
        if isinstance(c, Leaf):
            if c.type == 'name' and c.start_pos <= position <= c.end_pos:
                return c
        else:
            result = c.get_name_of_position(position)
            if result is not None:
                return result
    return None