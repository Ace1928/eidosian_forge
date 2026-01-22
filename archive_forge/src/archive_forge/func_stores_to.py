from pythran.analyses import LazynessAnalysis, StrictAliases, YieldPoints
from pythran.analyses import LocalNodeDeclarations, Immediates, RangeValues
from pythran.analyses import Ancestors
from pythran.config import cfg
from pythran.cxxtypes import TypeBuilder, ordered_set
from pythran.intrinsic import UserFunction, Class
from pythran.passmanager import ModuleAnalysis
from pythran.tables import operator_to_lambda, MODULES
from pythran.types.conversion import pytype_to_ctype
from pythran.types.reorder import Reorder
from pythran.utils import attr_to_path, cxxid, isnum, isextslice
from collections import defaultdict
from functools import reduce
import gast as ast
from itertools import islice
from copy import deepcopy
def stores_to(self, node):
    ancestors = self.ancestors[node] + (node,)
    stmt_indices = [i for i, n in enumerate(ancestors) if isinstance(n, (ast.Assign, ast.For))]
    if not stmt_indices:
        return True
    stmt_index = stmt_indices[-1]
    if isinstance(ancestors[stmt_index], ast.Assign):
        return ancestors[stmt_index + 1] is ancestors[stmt_index].value
    else:
        return ancestors[stmt_index + 1] is not ancestors[stmt_index].target