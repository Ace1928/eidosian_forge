import numpy as np
import pandas as pd
from ..doctools import document
from .stat import stat
@classmethod
def compute_group(cls, data, scales, **params):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(data[['x', 'y']], qhull_options=params['qhull_options'])
    idx = np.hstack([hull.vertices, hull.vertices[0]])
    new_data = pd.DataFrame({'x': data['x'].iloc[idx].to_numpy(), 'y': data['y'].iloc[idx].to_numpy(), 'area': hull.area})
    return new_data