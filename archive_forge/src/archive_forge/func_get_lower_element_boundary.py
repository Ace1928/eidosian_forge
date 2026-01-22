import logging
import bisect
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.set import SortedScalarSet
from pyomo.core.base.component import ModelComponentFactory
def get_lower_element_boundary(self, point):
    """Returns the first finite element point that is less than or
        equal to 'point'

        Parameters
        ----------
        point : `float`

        Returns
        -------
        float
        """
    if point in self._fe:
        if 'scheme' in self._discretization_info:
            if self._discretization_info['scheme'] == 'LAGRANGE-RADAU':
                tmp = self._fe.index(point)
                if tmp != 0:
                    return self._fe[tmp - 1]
        return point
    elif point < min(self._fe):
        logger.warning("The point '%s' is less than the lower bound of the ContinuousSet '%s'. Returning the lower bound " % (str(point), self.name))
        return min(self._fe)
    else:
        rev_fe = list(self._fe)
        rev_fe.reverse()
        for i in rev_fe:
            if i < point:
                return i