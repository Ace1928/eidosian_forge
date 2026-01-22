import numpy as np
from matplotlib import _api
from matplotlib.tri._triangulation import Triangulation
import matplotlib.tri._triinterpolate
def refine_triangulation(self, return_tri_index=False, subdiv=3):
    """
        Compute a uniformly refined triangulation *refi_triangulation* of
        the encapsulated :attr:`triangulation`.

        This function refines the encapsulated triangulation by splitting each
        father triangle into 4 child sub-triangles built on the edges midside
        nodes, recursing *subdiv* times.  In the end, each triangle is hence
        divided into ``4**subdiv`` child triangles.

        Parameters
        ----------
        return_tri_index : bool, default: False
            Whether an index table indicating the father triangle index of each
            point is returned.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles;
            hence, the default results in 64 refined subtriangles for each
            triangle of the initial triangulation.

        Returns
        -------
        refi_triangulation : `~matplotlib.tri.Triangulation`
            The refined triangulation.
        found_index : int array
            Index of the initial triangulation containing triangle, for each
            point of *refi_triangulation*.
            Returned only if *return_tri_index* is set to True.
        """
    refi_triangulation = self._triangulation
    ntri = refi_triangulation.triangles.shape[0]
    ancestors = np.arange(ntri, dtype=np.int32)
    for _ in range(subdiv):
        refi_triangulation, ancestors = self._refine_triangulation_once(refi_triangulation, ancestors)
    refi_npts = refi_triangulation.x.shape[0]
    refi_triangles = refi_triangulation.triangles
    if return_tri_index:
        found_index = np.full(refi_npts, -1, dtype=np.int32)
        tri_mask = self._triangulation.mask
        if tri_mask is None:
            found_index[refi_triangles] = np.repeat(ancestors, 3).reshape(-1, 3)
        else:
            ancestor_mask = tri_mask[ancestors]
            found_index[refi_triangles[ancestor_mask, :]] = np.repeat(ancestors[ancestor_mask], 3).reshape(-1, 3)
            found_index[refi_triangles[~ancestor_mask, :]] = np.repeat(ancestors[~ancestor_mask], 3).reshape(-1, 3)
        return (refi_triangulation, found_index)
    else:
        return refi_triangulation