import os
import numpy as np
from numpy.testing import assert_array_equal
from .. import nifti2
from ..nifti1 import Nifti1Extension, Nifti1Extensions, Nifti1Header, Nifti1PairHeader
from ..nifti2 import Nifti2Header, Nifti2Image, Nifti2Pair, Nifti2PairHeader
from ..testing import data_path
from . import test_nifti1 as tn1
def test_nifti12_conversion():
    shape = (2, 3, 4)
    dtype_type = np.int64
    ext1 = Nifti1Extension(6, b'My comment')
    ext2 = Nifti1Extension(6, b'Fresh comment')
    for in_type, out_type in ((Nifti1Header, Nifti2Header), (Nifti1PairHeader, Nifti2Header), (Nifti1PairHeader, Nifti2PairHeader), (Nifti2Header, Nifti1Header), (Nifti2PairHeader, Nifti1Header), (Nifti2PairHeader, Nifti1PairHeader)):
        in_hdr = in_type()
        in_hdr.set_data_shape(shape)
        in_hdr.set_data_dtype(dtype_type)
        in_hdr.extensions[:] = [ext1, ext2]
        out_hdr = out_type.from_header(in_hdr)
        assert out_hdr.get_data_shape() == shape
        assert out_hdr.get_data_dtype() == dtype_type
        assert in_hdr.extensions == out_hdr.extensions