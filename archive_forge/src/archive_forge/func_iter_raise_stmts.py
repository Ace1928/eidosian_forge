important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def iter_raise_stmts(self):
    """
        Returns a generator of `raise_stmt`. Includes raise statements inside try-except blocks
        """

    def scan(children):
        for element in children:
            if element.type == 'raise_stmt' or (element.type == 'keyword' and element.value == 'raise'):
                yield element
            if element.type in _RETURN_STMT_CONTAINERS:
                yield from scan(element.children)
    return scan(self.children)