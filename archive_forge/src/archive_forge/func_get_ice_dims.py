import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def get_ice_dims(csa_dict):
    dims = get_scalar(csa_dict, 'ICE_Dims')
    if dims is None:
        return None
    return dims.split('_')