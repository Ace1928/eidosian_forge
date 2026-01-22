from itertools import chain
from Bio.SearchIO._utils import allitems, optionalcascade, getattr_str
from ._base import _BaseSearchObject
from .hsp import HSP
@property
def fragments(self):
    """Access the HSPFragment objects contained in the Hit."""
    return list(chain(*self._items))