from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def get_scalar():
    """
    Generates a practice ScalarAxis axis with names ('one', 'two', 'three')

    Returns
    -------
    ScalarAxis axis
    """
    return axes.ScalarAxis(['one', 'two', 'three'])