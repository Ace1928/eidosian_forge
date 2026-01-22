from __future__ import annotations
from traitlets import Set, Unicode
from .base import Preprocessor
def check_cell_conditions(self, cell, resources, index):
    """
        Checks that a cell has a tag that is to be removed

        Returns: Boolean.
        True means cell should *not* be removed.
        """
    return not self.remove_cell_tags.intersection(cell.get('metadata', {}).get('tags', []))