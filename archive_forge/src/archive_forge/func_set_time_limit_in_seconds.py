import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def set_time_limit_in_seconds(self, limit: NumberT) -> None:
    """Sets a time limit for the solve() call."""
    self.__solve_helper.set_time_limit_in_seconds(limit)