import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
Calculate covariance assuming experimental observation errors are
                independent and follow a Gaussian
                distribution with constant variance.

                The formula used in parmest was verified against equations (7-5-15) and
                (7-5-16) in "Nonlinear Parameter Estimation", Y. Bard, 1974.

                This formula is also applicable if the objective is scaled by a constant;
                the constant cancels out. (was scaled by 1/n because it computes an
                expected value.)
                