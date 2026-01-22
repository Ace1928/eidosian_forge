from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def preprocess_targets(targets, instance, knownBlocks, gdp_tree=None):
    if gdp_tree is None:
        gdp_tree = get_gdp_tree(targets, instance, knownBlocks)
    return gdp_tree.reverse_topological_sort()