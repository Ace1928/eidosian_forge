import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def slice_indicator(self):
    """A number that is higher for higher slices in Z

        Comparing this number between two adjacent slices should give a
        difference equal to the voxel size in Z.

        See doc/theory/dicom_orientation for description
        """
    ipp = self.image_position
    s_norm = self.slice_normal
    if ipp is None or s_norm is None:
        return None
    return np.inner(ipp, s_norm)