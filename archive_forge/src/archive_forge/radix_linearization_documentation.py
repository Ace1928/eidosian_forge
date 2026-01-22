from pyomo.core.expr import ProductExpression, PowExpression
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Binary, value
from pyomo.core.base import (
from pyomo.core.base.var import _VarData
import logging

    This plugin generates linear relaxations of bilinear problems using
    the multiparametric disaggregation technique of Kolodziej, Castro,
    and Grossmann.  See:

    Scott Kolodziej, Pedro M. Castro, and Ignacio E. Grossmann. "Global
       optimization of bilinear programs with a multiparametric
       disaggregation technique."  J.Glob.Optim 57 pp.1039-1063. 2013.
       (DOI 10.1007/s10898-012-0022-1)
    