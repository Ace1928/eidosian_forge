import numpy as np
import h5py
from h5py import h5t
from .common import TestCase, ut
def test_custom_float_promotion(self):
    """Custom floats are correctly promoted to standard floats on read."""
    test_filename = self.mktemp().encode()
    dataset = b'DS1'
    dataset2 = b'DS2'
    dataset3 = b'DS3'
    dataset4 = b'DS4'
    dataset5 = b'DS5'
    dims = (4, 7)
    wdata = np.array([[-1.50066626e-09, 1.40062184e-09, 1.81216819e-10, 4.01087163e-10, 4.27917257e-10, -7.04858394e-11, 5.74800652e-10], [-1.50066626e-09, 4.86579665e-10, 3.42879503e-10, 5.12045517e-10, 5.10226528e-10, 2.24190444e-10, 3.93356459e-10], [-1.50066626e-09, 5.24778443e-10, 8.19454726e-10, 1.28966349e-09, 1.68483894e-10, 5.7127636e-11, -1.08684617e-10], [-1.50066626e-09, -1.08343556e-10, -1.58934199e-10, 8.52196536e-10, 6.18456397e-10, 6.16637408e-10, 1.31694833e-09]], dtype=np.float32)
    wdata2 = np.array([[-1.50066626e-09, 5.63886715e-10, -8.74251782e-11, 1.32558853e-10, 1.59161573e-10, 2.29420039e-10, -7.24185156e-11], [-1.50066626e-09, 1.87810656e-10, 7.74889486e-10, 3.95630195e-10, 9.42236511e-10, 8.38554115e-10, -8.71978045e-11], [-1.50066626e-09, 6.20275387e-10, 7.34871719e-10, 6.64840627e-10, 2.64662958e-10, 1.05319486e-09, 1.6825652e-10], [-1.50066626e-09, 1.67347025e-10, 5.12045517e-10, 3.3651304e-10, 1.02545528e-10, 1.2878445e-09, 4.06089384e-10]], dtype=np.float32)
    fid = h5py.h5f.create(test_filename)
    space = h5py.h5s.create_simple(dims)
    mytype = h5t.IEEE_F16LE.copy()
    mytype.set_fields(14, 9, 5, 0, 9)
    mytype.set_size(2)
    mytype.set_ebias(53)
    mytype.lock()
    dset = h5py.h5d.create(fid, dataset, mytype, space)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata)
    del dset
    mytype2 = h5t.IEEE_F16LE.copy()
    mytype2.set_fields(15, 9, 6, 0, 9)
    mytype2.set_size(2)
    mytype2.set_ebias(53)
    mytype2.lock()
    dset = h5py.h5d.create(fid, dataset2, mytype2, space)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)
    del dset
    mytype3 = h5t.IEEE_F16LE.copy()
    mytype3.set_fields(15, 10, 5, 0, 10)
    mytype3.set_size(2)
    mytype3.set_ebias(15)
    mytype3.lock()
    dset = h5py.h5d.create(fid, dataset3, mytype3, space)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)
    del dset
    mytype4 = h5t.IEEE_F16LE.copy()
    mytype4.set_fields(15, 10, 5, 0, 10)
    mytype4.set_size(2)
    mytype4.set_ebias(258)
    mytype4.lock()
    dset = h5py.h5d.create(fid, dataset4, mytype4, space)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)
    del dset
    dset = h5py.h5d.create(fid, dataset5, h5t.NATIVE_LDOUBLE, space)
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata2)
    del space
    del dset
    del fid
    f = h5py.File(test_filename, 'r')
    values = f[dataset][:]
    np.testing.assert_array_equal(values, wdata)
    self.assertEqual(values.dtype, np.dtype('<f4'))
    values = f[dataset2][:]
    np.testing.assert_array_equal(values, wdata2)
    self.assertEqual(values.dtype, np.dtype('<f4'))
    dset = f[dataset3]
    try:
        self.assertEqual(dset.dtype, np.dtype('<f2'))
    except AttributeError:
        self.assertEqual(dset.dtype, np.dtype('<f4'))
    dset = f[dataset4]
    self.assertEqual(dset.dtype, np.dtype('<f8'))
    dset = f[dataset5]
    self.assertEqual(dset.dtype, np.longdouble)