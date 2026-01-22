import logging
import sys
from weakref import ref as weakref_ref
import gc
import math
from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.opt.results import Solution, SolverStatus, UndefinedData
from contextlib import nullcontext
from io import StringIO
def compute_statistics(self, active=True):
    """
        Compute model statistics
        """
    self.statistics.number_of_variables = 0
    self.statistics.number_of_constraints = 0
    self.statistics.number_of_objectives = 0
    for block in self.block_data_objects(active=active):
        for data in block.component_map(Var, active=active).values():
            self.statistics.number_of_variables += len(data)
        for data in block.component_map(Objective, active=active).values():
            self.statistics.number_of_objectives += len(data)
        for data in block.component_map(Constraint, active=active).values():
            self.statistics.number_of_constraints += len(data)