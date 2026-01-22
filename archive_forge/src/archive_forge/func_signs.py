from .links import CrossingStrand
from ..graphs import CyclicList
def signs(self):
    return [C.sign() for C in self.crossings]