import numpy as np
from matplotlib import _api
@_api.deprecated('3.8')
def ortho_transformation(zfront, zback):
    return _ortho_transformation(zfront, zback)