import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def slice_normal(self):
    std_slice_normal = super().slice_normal
    csa_slice_normal = csar.get_slice_normal(self.csa_header)
    if std_slice_normal is None and csa_slice_normal is None:
        return None
    elif std_slice_normal is None:
        return np.array(csa_slice_normal)
    elif csa_slice_normal is None:
        return std_slice_normal
    else:
        dot_prod = np.dot(csa_slice_normal, std_slice_normal)
        assert np.allclose(np.fabs(dot_prod), 1.0, atol=1e-05)
        if dot_prod < 0:
            return -std_slice_normal
        else:
            return std_slice_normal