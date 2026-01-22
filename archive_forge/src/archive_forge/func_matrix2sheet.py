import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def matrix2sheet(self, float_row, float_col):
    """
        Convert a floating-point location (float_row,float_col) in
        matrix coordinates to its corresponding location (x,y) in
        sheet coordinates.

        Valid for scalar or array float_row and float_col.

        Inverse of sheet2matrix().
        """
    xoffset = float_col * self.__xstep
    if isinstance(self.lbrt[0], datetime_types):
        xoffset = np.timedelta64(int(round(xoffset)), self._time_unit)
    x = self.lbrt[0] + xoffset
    yoffset = float_row * self.__ystep
    if isinstance(self.lbrt[3], datetime_types):
        yoffset = np.timedelta64(int(round(yoffset)), self._time_unit)
    y = self.lbrt[3] - yoffset
    return (x, y)