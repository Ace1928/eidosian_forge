import logging
import bisect
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.set import SortedScalarSet
from pyomo.core.base.component import ModelComponentFactory
def get_discretization_info(self):
    """Returns a `dict` with information on the discretization scheme
        that has been applied to the :py:class:`ContinuousSet`.

        Returns
        -------
        `dict`
        """
    return self._discretization_info