import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline import convert
from nibabel.testing import get_test_data
@pytest.mark.parametrize('data_dtype', ('u1', 'i2', 'float32', 'float', 'int64'))
def test_convert_dtype(tmp_path, data_dtype):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / 'output.nii.gz'
    orig = nib.load(infile)
    assert not outfile.exists()
    expected_dtype = np.dtype(data_dtype).newbyteorder(orig.header.endianness)
    convert.main([str(infile), str(outfile), '--out-dtype', data_dtype])
    assert outfile.is_file()
    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.get_data_dtype() == expected_dtype