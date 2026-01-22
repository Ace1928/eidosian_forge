import types
from typing import Callable, Optional
from ortools.math_opt import parameters_pb2
from ortools.math_opt.core.python import solver
from ortools.math_opt.python import callback
from ortools.math_opt.python import compute_infeasible_subsystem_result
from ortools.math_opt.python import message_callback
from ortools.math_opt.python import model
from ortools.math_opt.python import model_parameters
from ortools.math_opt.python import parameters
from ortools.math_opt.python import result
from pybind11_abseil.status import StatusNotOk
Closes the solver.