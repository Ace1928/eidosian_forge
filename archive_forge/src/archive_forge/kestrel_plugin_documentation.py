import logging
import os
import re
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager
from pyomo.core.base import Block
import pyomo.neos.kestrel

        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        