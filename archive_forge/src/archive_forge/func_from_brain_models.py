from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
@classmethod
def from_brain_models(cls, named_brain_models):
    """
        Creates a Parcel axis from a list of BrainModelAxis axes with names

        Parameters
        ----------
        named_brain_models : iterable of 2-element tuples of string and BrainModelAxis
            list of (parcel name, brain model representation) pairs defining each parcel

        Returns
        -------
        ParcelsAxis
        """
    nparcels = len(named_brain_models)
    affine = None
    volume_shape = None
    all_names = []
    all_voxels = np.zeros(nparcels, dtype='object')
    all_vertices = np.zeros(nparcels, dtype='object')
    nvertices = {}
    for idx_parcel, (parcel_name, bm) in enumerate(named_brain_models):
        all_names.append(parcel_name)
        voxels = bm.voxel[bm.volume_mask]
        if voxels.shape[0] != 0:
            if affine is None:
                affine = bm.affine
                volume_shape = bm.volume_shape
            elif not np.allclose(affine, bm.affine) or volume_shape != bm.volume_shape:
                raise ValueError('Can not combine brain models defined in different volumes into a single Parcel axis')
        all_voxels[idx_parcel] = voxels
        vertices = {}
        for name, _, bm_part in bm.iter_structures():
            if name in bm.nvertices.keys():
                if name in nvertices.keys() and nvertices[name] != bm.nvertices[name]:
                    raise ValueError(f'Got multiple conflicting number of vertices for surface structure {name}')
                nvertices[name] = bm.nvertices[name]
                vertices[name] = bm_part.vertex
        all_vertices[idx_parcel] = vertices
    return ParcelsAxis(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)