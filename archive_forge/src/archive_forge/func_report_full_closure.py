import os
import stat
from itertools import filterfalse
from types import GenericAlias
def report_full_closure(self):
    self.report()
    for sd in self.subdirs.values():
        print()
        sd.report_full_closure()