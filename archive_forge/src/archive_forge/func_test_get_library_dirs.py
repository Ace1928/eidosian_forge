from unittest import TestCase
import zmq
def test_get_library_dirs(self):
    from os.path import basename
    libdirs = zmq.get_library_dirs()
    assert isinstance(libdirs, list)
    assert len(libdirs) == 1
    parent = libdirs[0]
    assert isinstance(parent, str)
    libdir = basename(parent)
    assert libdir == 'zmq'