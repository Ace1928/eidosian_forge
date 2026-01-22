from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Iterable
import numbers
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import expression
from cvxpy.settings import (
def save_value(self, val) -> None:
    self._value = val