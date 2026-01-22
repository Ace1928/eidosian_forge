from numpy.testing import (assert_, assert_allclose, assert_equal,
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import gcrotmk, gmres
Tests for the linalg._isolve.gcrotmk module
