import numpy as np
from matplotlib import _api
@_api.deprecated('3.8')
def proj_trans_points(points, M):
    return _proj_trans_points(points, M)