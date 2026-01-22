import os
import sys
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.core.base import Objective, Var, Constraint, value, ConcreteModel
def pyomo2nl(args=None):
    from pyomo.scripting.pyomo_main import main
    if args is None:
        return main()
    else:
        return main(['convert', '--format=nl'] + args)