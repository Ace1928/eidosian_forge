import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def solve_time(self) -> datetime.timedelta:
    """Shortcut for SolveResult.solve_stats.solve_time."""
    return self.solve_stats.solve_time