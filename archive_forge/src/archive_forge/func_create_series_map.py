import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def create_series_map(applies_to_matrix_dimension):
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_SERIES', number_of_series_points=13, series_exponent=-3, series_start=18.2, series_step=10.5, series_unit='SECOND')