from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def get_next_sibling(self):
    """
        Returns the node immediately following this node in this parent's
        children list. If this node does not have a next sibling, it is None
        """
    parent = self.parent
    if parent is None:
        return None
    for i, child in enumerate(parent.children):
        if child is self:
            try:
                return self.parent.children[i + 1]
            except IndexError:
                return None