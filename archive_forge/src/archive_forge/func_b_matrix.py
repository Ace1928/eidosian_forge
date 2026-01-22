import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def b_matrix(self):
    """Get DWI B matrix referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        B : (3,3) array or None
           B matrix in *voxel* orientation space.  Returns None if this is
           not a Siemens header with the required information.  We return
           None if this is a b0 acquisition
        """
    hdr = self.csa_header
    B = csar.get_b_matrix(hdr)
    if B is None:
        bval_requested = csar.get_b_value(hdr)
        if bval_requested is None:
            return None
        if bval_requested != 0:
            raise csar.CSAError('No B matrix and b value != 0')
        return np.zeros((3, 3))
    R = self.rotation_matrix.T
    B_vox = np.dot(R, np.dot(B, R.T))
    return nearest_pos_semi_def(B_vox)