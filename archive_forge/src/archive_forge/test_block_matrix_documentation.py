import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sp
from scipy.sparse import coo_matrix, bmat
from pyomo.contrib.pynumero.sparse import (
import warnings

        Test

        [A  B  C   *  [G  J   = [A*G + B*H + C*I    A*J + B*K + C*L
         D  E  F]      H  K      D*G + E*H + F*I    D*J + E*K + F*L]
                       I  L]
        