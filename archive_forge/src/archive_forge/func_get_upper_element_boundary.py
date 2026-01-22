import logging
import bisect
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.set import SortedScalarSet
from pyomo.core.base.component import ModelComponentFactory
def get_upper_element_boundary(self, point):
    """Returns the first finite element point that is greater or equal
        to 'point'

        Parameters
        ----------
        point : `float`

        Returns
        -------
        float
        """
    if point in self._fe:
        return point
    elif point > max(self._fe):
        logger.warning("The point '%s' exceeds the upper bound of the ContinuousSet '%s'. Returning the upper bound" % (str(point), self.name))
        return max(self._fe)
    else:
        for i in self._fe:
            if i > point:
                return i