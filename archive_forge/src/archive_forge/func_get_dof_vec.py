import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
@staticmethod
def get_dof_vec(tri_z, tri_dz, J):
    """
        Compute the dof vector of a triangle, from the value of f, df and
        of the local Jacobian at each node.

        Parameters
        ----------
        tri_z : shape (3,) array
            f nodal values.
        tri_dz : shape (3, 2) array
            df/dx, df/dy nodal values.
        J
            Jacobian matrix in local basis of apex 0.

        Returns
        -------
        dof : shape (9,) array
            For each apex ``iapex``::

                dof[iapex*3+0] = f(Ai)
                dof[iapex*3+1] = df(Ai).(AiAi+)
                dof[iapex*3+2] = df(Ai).(AiAi-)
        """
    npt = tri_z.shape[0]
    dof = np.zeros([npt, 9], dtype=np.float64)
    J1 = _ReducedHCT_Element.J0_to_J1 @ J
    J2 = _ReducedHCT_Element.J0_to_J2 @ J
    col0 = J @ np.expand_dims(tri_dz[:, 0, :], axis=2)
    col1 = J1 @ np.expand_dims(tri_dz[:, 1, :], axis=2)
    col2 = J2 @ np.expand_dims(tri_dz[:, 2, :], axis=2)
    dfdksi = _to_matrix_vectorized([[col0[:, 0, 0], col1[:, 0, 0], col2[:, 0, 0]], [col0[:, 1, 0], col1[:, 1, 0], col2[:, 1, 0]]])
    dof[:, 0:7:3] = tri_z
    dof[:, 1:8:3] = dfdksi[:, 0]
    dof[:, 2:9:3] = dfdksi[:, 1]
    return dof