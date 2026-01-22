from pyomo.common.dependencies import numpy as np, numpy_available, pandas_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments, MeasurementVariables, DesignVariables
from pyomo.environ import value, ConcreteModel
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.opt import SolverFactory
Test the kinetics example with both the sequential_finite mode and the direct_kaug mode