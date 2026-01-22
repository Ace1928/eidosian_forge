import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def compute_dof_from_df(self):
    """
        Compute reduced-HCT elements degrees of freedom, from the gradient.
        """
    J = CubicTriInterpolator._get_jacobian(self._tris_pts)
    tri_z = self.z[self._triangles]
    tri_dz = self.dz[self._triangles]
    tri_dof = self.get_dof_vec(tri_z, tri_dz, J)
    return tri_dof