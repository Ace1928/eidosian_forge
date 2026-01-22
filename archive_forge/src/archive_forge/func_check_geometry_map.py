import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def check_geometry_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS'
    assert len(list(mapping.brain_models)) == 3
    left_thalamus, left_cortex, right_cortex = mapping.brain_models
    assert left_thalamus.index_offset == 0
    assert left_thalamus.index_count == 4
    assert left_thalamus.model_type == 'CIFTI_MODEL_TYPE_VOXELS'
    assert left_thalamus.brain_structure == brain_models[0][0]
    assert left_thalamus.vertex_indices is None
    assert left_thalamus.surface_number_of_vertices is None
    assert left_thalamus.voxel_indices_ijk._indices == brain_models[0][1]
    assert left_cortex.index_offset == 4
    assert left_cortex.index_count == 5
    assert left_cortex.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
    assert left_cortex.brain_structure == brain_models[1][0]
    assert left_cortex.voxel_indices_ijk is None
    assert left_cortex.vertex_indices._indices == brain_models[1][1]
    assert left_cortex.surface_number_of_vertices == number_of_vertices
    assert right_cortex.index_offset == 9
    assert right_cortex.index_count == 1
    assert right_cortex.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
    assert right_cortex.brain_structure == brain_models[2][0]
    assert right_cortex.voxel_indices_ijk is None
    assert right_cortex.vertex_indices._indices == brain_models[2][1]
    assert right_cortex.surface_number_of_vertices == number_of_vertices
    assert mapping.volume.volume_dimensions == dimensions
    assert (mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all()