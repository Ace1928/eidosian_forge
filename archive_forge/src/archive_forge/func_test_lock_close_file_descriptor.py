import pytest
from ase.utils import Lock
def test_lock_close_file_descriptor(tmp_path):
    """Test that lock file descriptor is properly closed."""
    lock = Lock(tmp_path / 'lockfile', timeout=1.0)
    with lock:
        assert not lock.fd.closed
    assert lock.fd.closed