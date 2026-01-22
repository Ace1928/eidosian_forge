from .links import CrossingStrand
from ..graphs import CyclicList
def make_consistent(self):
    sign = self.crossings[0].sign()
    for C in self.crossings:
        if C.sign() != sign:
            C.swap_crossing()