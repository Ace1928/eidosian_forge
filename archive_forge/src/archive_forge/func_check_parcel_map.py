import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def check_parcel_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_PARCELS'
    assert len(list(mapping.parcels)) == len(parcels)
    for (name, elements), parcel in zip(parcels, mapping.parcels):
        assert parcel.name == name
        idx_surface = 0
        for element in elements:
            if isinstance(element[0], str):
                surface = parcel.vertices[idx_surface]
                assert surface.brain_structure == element[0]
                assert surface._vertices == element[1]
                idx_surface += 1
            else:
                assert parcel.voxel_indices_ijk._indices == element
    for surface, orientation in zip(mapping.surfaces, ('LEFT', 'RIGHT')):
        assert surface.brain_structure == f'CIFTI_STRUCTURE_CORTEX_{orientation}'
        assert surface.surface_number_of_vertices == number_of_vertices
    assert mapping.volume.volume_dimensions == dimensions
    assert (mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all()