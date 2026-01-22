from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def get_parcels():
    """
    Generates a practice Parcel axis out of all practice brain models

    Returns
    -------
    Parcel axis
    """
    bml = list(get_brain_models())
    return axes.ParcelsAxis.from_brain_models([('mixed', bml[0] + bml[2]), ('volume', bml[1]), ('surface', bml[3])])