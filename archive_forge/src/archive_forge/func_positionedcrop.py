import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def positionedcrop(self, x, y, sheet_coord_system):
    """
        Offset the bounds_template to this cf's location and store the
        result in the 'bounds' attribute.

        Also stores the input_sheet_slice for access by C.
        """
    cf_row, cf_col = sheet_coord_system.sheet2matrixidx(x, y)
    bounds_x, bounds_y = self.compute_bounds(sheet_coord_system).centroid()
    b_row, b_col = sheet_coord_system.sheet2matrixidx(bounds_x, bounds_y)
    row_offset = cf_row - b_row
    col_offset = cf_col - b_col
    self.translate(row_offset, col_offset)