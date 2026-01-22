import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
A suite of basic tests for implicit function solvers.

    A "concrete" subclass should be defined for each implicit function
    solver. This subclass should implement get_solver_class, then
    add "test" methods that call the following methods:

        _test_implicit_function_1
        _test_implicit_function_inputs_dont_appear
        _test_implicit_function_no_inputs
        _test_implicit_function_with_extra_variables

    These methods are private so they don't get picked up on the base
    class by pytest.

    