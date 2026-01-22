import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def q_vector(self):
    """Get DWI q vector referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        q: (3,) array
           Estimated DWI q vector in *voxel* orientation space.  Returns
           None if this is not (detectably) a DWI
        """
    B = self.b_matrix
    if B is None:
        return None
    return B2q(B, tol=1e-08)