from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def insert_cup(self, i):
    """
        Insert an new matching at (i, i + 1)
        """
    return VElement({insert_cup(m, i): c for m, c in self.dict.items()})