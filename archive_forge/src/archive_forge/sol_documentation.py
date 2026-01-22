import re
from pyomo.opt.base import results
from pyomo.opt.base.formats import ResultsFormat
from pyomo.opt import SolverResults, SolutionStatus, SolverStatus, TerminationCondition

        Parse a *.sol file
        