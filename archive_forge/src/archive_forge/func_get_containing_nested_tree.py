from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def get_containing_nested_tree(self, path):
    """Find the nested tree that contains a path.

        Returns: tuple with (nested tree and path inside the nested tree)
        """
    for nested_path in self.iter_references():
        nested_path += '/'
        if path.startswith(nested_path):
            nested_tree = self.get_nested_tree(nested_path)
            return (nested_tree, path[len(nested_path):])
    else:
        return (None, None)