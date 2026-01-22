import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
def check_hcp_grayordinates(brain_model):
    """Checks that a BrainModelAxis matches the expected 32k HCP grayordinates"""
    assert isinstance(brain_model, cifti2_axes.BrainModelAxis)
    structures = list(brain_model.iter_structures())
    assert len(structures) == len(hcp_labels)
    idx_start = 0
    for idx, (name, _, bm), label, nel in zip(range(len(structures)), structures, hcp_labels, hcp_n_elements):
        if idx < 2:
            assert name in bm.nvertices.keys()
            assert (bm.voxel == -1).all()
            assert (bm.vertex != -1).any()
            assert bm.nvertices[name] == 32492
        else:
            assert name not in bm.nvertices.keys()
            assert (bm.voxel != -1).any()
            assert (bm.vertex == -1).all()
            assert (bm.affine == hcp_affine).all()
            assert bm.volume_shape == (91, 109, 91)
        assert name == cifti2_axes.BrainModelAxis.to_cifti_brain_structure_name(label)
        assert len(bm) == nel
        assert (bm.name == brain_model.name[idx_start:idx_start + nel]).all()
        assert (bm.voxel == brain_model.voxel[idx_start:idx_start + nel]).all()
        assert (bm.vertex == brain_model.vertex[idx_start:idx_start + nel]).all()
        idx_start += nel
    assert idx_start == len(brain_model)
    assert (brain_model.vertex[:5] == np.arange(5)).all()
    assert structures[0][2].vertex[-1] == 32491
    assert structures[1][2].vertex[0] == 0
    assert structures[1][2].vertex[-1] == 32491
    assert structures[-1][2].name[-1] == brain_model.name[-1]
    assert (structures[-1][2].voxel[-1] == brain_model.voxel[-1]).all()
    assert structures[-1][2].vertex[-1] == brain_model.vertex[-1]
    assert (brain_model.voxel[-1] == [38, 55, 46]).all()
    assert (brain_model.voxel[70000] == [56, 22, 19]).all()