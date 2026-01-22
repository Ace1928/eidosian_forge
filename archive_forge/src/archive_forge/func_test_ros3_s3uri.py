import h5py
from h5py._hl.files import make_fapl
import pytest
@pytest.mark.nonetwork
def test_ros3_s3uri():
    """Use S3 URI with ROS3 driver"""
    with h5py.File('s3://dandiarchive/ros3test.hdf5', 'r', driver='ros3', aws_region=b'us-east-2') as f:
        assert f
        assert 'mydataset' in f.keys()
        assert f['mydataset'].shape == (100,)