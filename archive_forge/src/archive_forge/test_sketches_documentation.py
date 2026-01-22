import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm

    Testing the Clarkson Woodruff Transform
    