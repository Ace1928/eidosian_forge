import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear

Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
