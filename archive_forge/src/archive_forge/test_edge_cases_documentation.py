import numpy as np
import pytest
from opt_einsum import contract, contract_expression

Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
