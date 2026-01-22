from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (

        Continuing from the non-parametrized example, instead of a pure variable
        input, we take a variable that has been multiplied elementwise by a parameter.

        The trace of this expression is then given by

            x11  x21  x12  x22
        [
            [[1   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   1]]
        ]
        