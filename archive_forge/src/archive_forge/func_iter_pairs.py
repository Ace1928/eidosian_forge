import re
from .. import osutils
from ..iterablefile import IterableFile
def iter_pairs(self):
    """Return iterator of tag, value pairs."""
    return iter(self.items)