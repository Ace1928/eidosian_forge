import numpy as np
import pytest
import cvxpy as cvx
import cvxpy.problems.iterative as iterative
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import prune_constants
from cvxpy.tests.base_test import BaseTest
Convolution with 0D input.
        