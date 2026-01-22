important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def is_definition(self, include_setitem=False):
    """
        Returns True if the name is being defined.
        """
    return self.get_definition(include_setitem=include_setitem) is not None