from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor

    This is a unit test for the same problem.
    The variable and parameter namings are derived from the problem above.
    