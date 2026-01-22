important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
@property
def star_count(self):
    """
        Is `0` in case of `foo`, `1` in case of `*foo` or `2` in case of
        `**foo`.
        """
    first = self.children[0]
    if first in ('*', '**'):
        return len(first.value)
    return 0