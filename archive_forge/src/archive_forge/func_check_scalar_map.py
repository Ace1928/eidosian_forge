import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def check_scalar_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_SCALARS'
    assert len(list(mapping.named_maps)) == 2
    for expected, named_map in zip(scalars, mapping.named_maps):
        assert named_map.map_name == expected[0]
        if len(expected[1]) == 0:
            assert named_map.metadata is None
        else:
            assert named_map.metadata == expected[1]