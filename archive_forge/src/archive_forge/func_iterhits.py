from copy import deepcopy
from itertools import chain
from Bio.SearchIO._utils import optionalcascade
from ._base import _BaseSearchObject
from .hit import Hit
def iterhits(self):
    """Return an iterator over the Hit objects."""
    yield from self._items.values()