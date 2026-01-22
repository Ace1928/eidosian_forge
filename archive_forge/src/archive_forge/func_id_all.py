from itertools import chain
from Bio.SearchIO._utils import allitems, optionalcascade, getattr_str
from ._base import _BaseSearchObject
from .hsp import HSP
@property
def id_all(self):
    """Alternative ID(s) of the Hit."""
    return [self.id] + self._id_alt