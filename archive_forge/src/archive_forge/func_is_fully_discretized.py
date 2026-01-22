import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
def is_fully_discretized(self):
    """
        Check to see if all the
        :py:class:`ContinuousSets<pyomo.dae.ContinuousSet>` this derivative
        is taken with respect to have been discretized.

        Returns
        -------
        `boolean`
        """
    for i in self._wrt:
        if 'scheme' not in i.get_discretization_info():
            return False
    return True