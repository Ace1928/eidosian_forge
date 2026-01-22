import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
This helper takes an ExternalGreyBoxModel and provides the residual,
        Jacobian, and Hessian computation mapped to the correct variable space.

        The ExternalGreyBoxModel provides an interface that supports
        equality constraints (pure residuals) and output equations. Let
        u be the inputs, o be the outputs, and x be the full set of
        primal variables from the entire pyomo_nlp.

        With this, the ExternalGreyBoxModel provides the residual
        computations w_eq(u), and w_o(u), as well as the Jacobians,
        Jw_eq(u), and Jw_o(u). This helper provides h(x)=0, where h(x) =
        [h_eq(x); h_o(x)-o] and h_eq(x)=w_eq(Pu*x), and
        h_o(x)=w_o(Pu*x), and Pu is a mapping from the full primal
        variables "x" to the inputs "u".

        It also provides the Jacobian of h w.r.t. x.
           J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]
        where Po is a mapping from the full primal variables "x" to the
        outputs "o".

        