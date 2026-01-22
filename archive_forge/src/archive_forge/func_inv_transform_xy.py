import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
def inv_transform_xy(self, x, y):
    return self._aux_transform.inverted().transform(np.column_stack([x, y])).T