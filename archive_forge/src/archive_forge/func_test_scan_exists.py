import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_scan_exists():
    assert scan_1.exists()
    assert isinstance(scan_1, object)
    assert isinstance(scan_1, pyxnat.core.resources.Scan)
    assert str(scan_1) != '<Scan Object> JKL'