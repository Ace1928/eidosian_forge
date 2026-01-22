import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def unionify(self, other):
    for k, v in other.items():
        if k in self.result:
            self.result[k] = self.result[k].union(v)
        else:
            self.result[k] = v