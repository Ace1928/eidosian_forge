import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def image_orient_patient(self):
    """
        Note that this is _not_ LR flipped
        """
    try:
        iop = self.shared.PlaneOrientationSequence[0].ImageOrientationPatient
    except AttributeError:
        try:
            iop = self.frames[0].PlaneOrientationSequence[0].ImageOrientationPatient
        except AttributeError:
            raise WrapperError('Not enough information for image_orient_patient')
    if iop is None:
        return None
    iop = np.array(list(map(float, iop)))
    return np.array(iop).reshape(2, 3).T