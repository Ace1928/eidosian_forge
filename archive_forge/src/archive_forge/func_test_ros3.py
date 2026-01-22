import h5py
from h5py._hl.files import make_fapl
import pytest
@pytest.mark.nonetwork
def test_ros3():
    """ ROS3 driver and options """
    with h5py.File('https://dandiarchive.s3.amazonaws.com/ros3test.hdf5', 'r', driver='ros3') as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f['mydataset'].shape == (100,)