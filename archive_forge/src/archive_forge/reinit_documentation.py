import logging
import random
from pyomo.core import Var
Reinitializes all variable values in the model.

    Excludes fixed, noncontinuous, and unbounded variables.

    