import itertools
import logging
import math
from io import StringIO
from contextlib import nullcontext
from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
def mutable_param_gen(b):
    for param in block.component_objects(Param):
        if param.mutable and param.is_indexed():
            param_data_iter = (param_data for index, param_data in param.items())
        elif not param.is_indexed():
            param_data_iter = iter([param])
        else:
            param_data_iter = iter([])
        for param_data in param_data_iter:
            yield param_data