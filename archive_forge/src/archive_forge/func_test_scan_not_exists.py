import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_scan_not_exists():
    assert not scan_2.exists()
    assert isinstance(scan_2, object)
    assert isinstance(scan_2, pyxnat.core.resources.Scan)
    assert str(scan_2) == '<Scan Object> JKL'