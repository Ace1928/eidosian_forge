import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def check_series_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_SERIES'
    assert mapping.number_of_series_points == 13
    assert mapping.series_exponent == -3
    assert mapping.series_start == 18.2
    assert mapping.series_step == 10.5
    assert mapping.series_unit == 'SECOND'