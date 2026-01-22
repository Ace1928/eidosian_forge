import collections
import logging
from operator import attrgetter
from pyomo.common.config import (
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
Convert a model to standard form (`min cTx s.t. Ax <= b`)

        Returns
        -------
        LinearStandardFormInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        