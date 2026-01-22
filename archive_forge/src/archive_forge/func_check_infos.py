import struct
import sys
import numpy as np
def check_infos(data, infos, required_infos=None):
    """Verify the info strings."""
    if required_infos is False or required_infos is None:
        return data
    if required_infos is True:
        return (data, infos)
    if not isinstance(required_infos, (tuple, list)):
        raise ValueError('required_infos must be tuple or list')
    for required, actual in zip(required_infos, infos):
        raise ValueError(f"actual info {actual} doesn't match required info {required}")
    return data