import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def reduce_collocation_points(self, instance, var=None, ncp=None, contset=None):
    """
        This method will add additional constraints to a model to reduce the
        number of free collocation points (degrees of freedom) for a particular
        variable.

        Parameters
        ----------
        instance : Pyomo model
            The discretized Pyomo model to add constraints to

        var : ``pyomo.environ.Var``
            The Pyomo variable for which the degrees of freedom will be reduced

        ncp : int
            The new number of free collocation points for `var`. Must be
            less that the number of collocation points used in discretizing
            the model.

        contset : ``pyomo.dae.ContinuousSet``
            The :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that was
            discretized and for which the `var` will have a reduced number
            of degrees of freedom

        """
    if contset is None:
        raise TypeError("A continuous set must be specified using the keyword 'contset'")
    if contset.ctype is not ContinuousSet:
        raise TypeError("The component specified using the 'contset' keyword must be a ContinuousSet")
    ds = contset
    if len(self._ncp) == 0:
        raise RuntimeError('This method should only be called after using the apply() method to discretize the model')
    elif None in self._ncp:
        tot_ncp = self._ncp[None]
    elif ds.name in self._ncp:
        tot_ncp = self._ncp[ds.name]
    else:
        raise ValueError("ContinuousSet '%s' has not been discretized, please call the apply_to() method with this ContinuousSet to discretize it before calling this method" % ds.name)
    if var is None:
        raise TypeError('A variable must be specified')
    if var.ctype is not Var:
        raise TypeError("The component specified using the 'var' keyword must be a variable")
    if ncp is None:
        raise TypeError('The number of collocation points must be specified')
    if ncp <= 0:
        raise ValueError('The number of collocation points must be at least 1')
    if ncp > tot_ncp:
        raise ValueError('The number of collocation points used to interpolate an individual variable must be less than the number used to discretize the original model')
    if ncp == tot_ncp:
        return instance
    if var.dim() == 0:
        raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
    varidx = var.index_set()
    if not varidx.subsets():
        if ds is not varidx:
            raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
    elif ds not in varidx.subsets():
        raise IndexError("ContinuousSet '%s' is not an indexing set of the variable '%s'" % (ds.name, var.name))
    if var.name in self._reduced_cp:
        temp = self._reduced_cp[var.name]
        if ds.name in temp:
            raise RuntimeError("Variable '%s' has already been constrained to a reduced number of collocation points over ContinuousSet '%s'.")
        else:
            temp[ds.name] = ncp
    else:
        self._reduced_cp[var.name] = {ds.name: ncp}
    list_name = var.local_name + '_interpolation_constraints'
    instance.add_component(list_name, ConstraintList())
    conlist = instance.find_component(list_name)
    t = list(ds)
    fe = ds._fe
    info = get_index_information(var, ds)
    tmpidx = info['non_ds']
    idx = info['index function']
    for n in tmpidx:
        for i in range(0, len(fe) - 1):
            for k in range(1, tot_ncp - ncp + 1):
                if ncp == 1:
                    conlist.add(var[idx(n, i, k)] == var[idx(n, i, tot_ncp)])
                else:
                    tmp = ds.ord(fe[i]) - 1
                    tmp2 = ds.ord(fe[i + 1]) - 1
                    ti = t[tmp + k]
                    tfit = t[tmp2 - ncp + 1:tmp2 + 1]
                    coeff = self._interpolation_coeffs(ti, tfit)
                    conlist.add(var[idx(n, i, k)] == sum((var[idx(n, i, j)] * next(coeff) for j in range(tot_ncp - ncp + 1, tot_ncp + 1))))
    return instance