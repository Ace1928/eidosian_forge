import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_sdpa_sdp_1(self) -> None:
    StandardTestSDPs.test_sdp_1min(solver='SDPA')
    StandardTestSDPs.test_sdp_1max(solver='SDPA')