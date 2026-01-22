import os
import stat
from itertools import filterfalse
from types import GenericAlias
def phase4_closure(self):
    self.phase4()
    for sd in self.subdirs.values():
        sd.phase4_closure()