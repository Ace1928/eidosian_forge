important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_defined_names(self, include_setitem=False):
    """
        Returns the a list of `Name` that the comprehension defines.
        """
    return _defined_names(self.children[1], include_setitem)