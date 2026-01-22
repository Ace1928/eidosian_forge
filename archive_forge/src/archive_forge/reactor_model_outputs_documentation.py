import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

This file contains a black box model representing a simple
reactor design problem described in the Pyomo book.
It is part of the external_grey_box example with PyNumero.

These functions solve a reactor model using scipy
Note: In this case, this model can be solved using
standard Pyomo constructs (see the Pyomo book), but
this is included as an example of the external grey
box model interface.
