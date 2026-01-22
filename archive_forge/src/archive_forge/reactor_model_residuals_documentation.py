import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

This file contains an external grey box model representing a simple
reactor design problem described in the Pyomo book.
It is part of the external_grey_box examples with PyNumero.

Note: In this case, this model can be solved using
standard Pyomo constructs (see the Pyomo book), but
this is included as an example of the external grey
box model interface.
