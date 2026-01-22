from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def test_brain_models():
    """
    Tests the introspection and creation of CIFTI-2 BrainModelAxis axes
    """
    bml = list(get_brain_models())
    assert len(bml[0]) == 3
    assert (bml[0].vertex == -1).all()
    assert (bml[0].voxel == [[0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert bml[0][1][0] == 'CIFTI_MODEL_TYPE_VOXELS'
    assert (bml[0][1][1] == [0, 4, 0]).all()
    assert bml[0][1][2] == axes.BrainModelAxis.to_cifti_brain_structure_name('thalamus_right')
    assert len(bml[1]) == 4
    assert (bml[1].vertex == -1).all()
    assert (bml[1].voxel == [[0, 0, 0], [0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert len(bml[2]) == 3
    assert (bml[2].voxel == -1).all()
    assert (bml[2].vertex == [0, 5, 10]).all()
    assert bml[2][1] == ('CIFTI_MODEL_TYPE_SURFACE', 5, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    assert len(bml[3]) == 4
    assert (bml[3].voxel == -1).all()
    assert (bml[3].vertex == [0, 5, 10, 13]).all()
    assert bml[4][1] == ('CIFTI_MODEL_TYPE_SURFACE', 9, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    assert len(bml[4]) == 3
    assert (bml[4].voxel == -1).all()
    assert (bml[4].vertex == [2, 9, 14]).all()
    for bm, label, is_surface in zip(bml, ['ThalamusRight', 'Other', 'cortex_left', 'Other'], (False, False, True, True)):
        assert np.all(bm.surface_mask == ~bm.volume_mask)
        structures = list(bm.iter_structures())
        assert len(structures) == 1
        name = structures[0][0]
        assert name == axes.BrainModelAxis.to_cifti_brain_structure_name(label)
        if is_surface:
            assert bm.nvertices[name] == 15
        else:
            assert name not in bm.nvertices
            assert (bm.affine == rand_affine).all()
            assert bm.volume_shape == vol_shape
    bmt = bml[0] + bml[1] + bml[2]
    assert len(bmt) == 10
    structures = list(bmt.iter_structures())
    assert len(structures) == 3
    for bm, (name, _, bm_split) in zip(bml[:3], structures):
        assert bm == bm_split
        assert (bm_split.name == name).all()
        assert bm == bmt[bmt.name == bm.name[0]]
        assert bm == bmt[np.where(bmt.name == bm.name[0])]
    bmt = bmt + bml[2]
    assert len(bmt) == 13
    structures = list(bmt.iter_structures())
    assert len(structures) == 3
    assert len(structures[-1][2]) == 6
    bmt.affine = np.eye(4)
    with pytest.raises(ValueError):
        bmt.affine = np.eye(3)
    with pytest.raises(ValueError):
        bmt.affine = np.eye(4).flatten()
    bmt.volume_shape = (5, 3, 1)
    with pytest.raises(ValueError):
        bmt.volume_shape = (5.0, 3, 1)
    with pytest.raises(ValueError):
        bmt.volume_shape = (5, 3, 1, 4)
    with pytest.raises(IndexError):
        bmt['thalamus_left']
    bm_vox = axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    assert np.all(bm_vox.name == ['CIFTI_STRUCTURE_THALAMUS_LEFT'] * 5)
    assert np.array_equal(bm_vox.vertex, np.full(5, -1))
    assert np.array_equal(bm_vox.voxel, np.full((5, 3), 1))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), volume_shape=(2, 3, 4))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('random_name', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('thalamus_left', voxel=-np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('thalamus_left', affine=np.eye(4), volume_shape=(2, 3, 4))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 2), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    bm_vertex = axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 20})
    assert np.array_equal(bm_vertex.name, ['CIFTI_STRUCTURE_CORTEX_LEFT'] * 5)
    assert np.array_equal(bm_vertex.vertex, np.full(5, 1))
    assert np.array_equal(bm_vertex.voxel, np.full((5, 3), -1))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_right': 20})
    with pytest.raises(ValueError):
        axes.BrainModelAxis('cortex_left', vertex=-np.ones(5, dtype=int), nvertices={'cortex_left': 20})
    with pytest.raises(ValueError):
        axes.BrainModelAxis.from_mask(np.arange(5) > 2, affine=np.ones(5))
    with pytest.raises(ValueError):
        axes.BrainModelAxis.from_mask(np.ones((5, 3)))
    bm_vox = axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    bm_vox + bm_vox
    assert (bm_vertex + bm_vox)[:bm_vertex.size] == bm_vertex
    assert (bm_vox + bm_vertex)[:bm_vox.size] == bm_vox
    for bm_added in (bm_vox + bm_vertex, bm_vertex + bm_vox):
        assert bm_added.nvertices == bm_vertex.nvertices
        assert np.all(bm_added.affine == bm_vox.affine)
        assert bm_added.volume_shape == bm_vox.volume_shape
    axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_vox)])
    with pytest.raises(Exception):
        bm_vox + get_label()
    bm_other_shape = axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(4, 3, 4))
    with pytest.raises(ValueError):
        bm_vox + bm_other_shape
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_other_shape)])
    bm_other_affine = axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4) * 2, volume_shape=(2, 3, 4))
    with pytest.raises(ValueError):
        bm_vox + bm_other_affine
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_other_affine)])
    bm_vertex = axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 20})
    bm_other_number = axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 30})
    with pytest.raises(ValueError):
        bm_vertex + bm_other_number
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vertex), ('b', bm_other_number)])
    bm_vox = axes.BrainModelAxis('thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(2, 3, 4))
    bm_other = deepcopy(bm_vox)
    assert bm_vox == bm_other
    bm_other.voxel[1, 0] = 0
    assert bm_vox != bm_other
    bm_other = deepcopy(bm_vox)
    bm_other.vertex[1] = 10
    assert bm_vox == bm_other, 'vertices are ignored in volumetric BrainModelAxis'
    bm_other = deepcopy(bm_vox)
    bm_other.name[1] = 'BRAIN_STRUCTURE_OTHER'
    assert bm_vox != bm_other
    bm_other = deepcopy(bm_vox)
    bm_other.affine[0, 0] = 10
    assert bm_vox != bm_other
    bm_other = deepcopy(bm_vox)
    bm_other.affine = None
    assert bm_vox != bm_other
    assert bm_other != bm_vox
    bm_other = deepcopy(bm_vox)
    bm_other.volume_shape = (10, 3, 4)
    assert bm_vox != bm_other
    bm_vertex = axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 20})
    bm_other = deepcopy(bm_vertex)
    assert bm_vertex == bm_other
    bm_other.voxel[1, 0] = 0
    assert bm_vertex == bm_other, 'voxels are ignored in surface BrainModelAxis'
    bm_other = deepcopy(bm_vertex)
    bm_other.vertex[1] = 10
    assert bm_vertex != bm_other
    bm_other = deepcopy(bm_vertex)
    bm_other.name[1] = 'BRAIN_STRUCTURE_CORTEX_RIGHT'
    assert bm_vertex != bm_other
    bm_other = deepcopy(bm_vertex)
    bm_other.nvertices['BRAIN_STRUCTURE_CORTEX_LEFT'] = 50
    assert bm_vertex != bm_other
    bm_other = deepcopy(bm_vertex)
    bm_other.nvertices['BRAIN_STRUCTURE_CORTEX_RIGHT'] = 20
    assert bm_vertex != bm_other
    assert bm_vox != get_parcels()
    assert bm_vertex != get_parcels()