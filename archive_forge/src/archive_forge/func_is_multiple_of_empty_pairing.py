from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def is_multiple_of_empty_pairing(self):
    return len(self.dict) == 1 and PerfectMatching([]) in self.dict