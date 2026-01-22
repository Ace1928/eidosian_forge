import pyxnat.core.resources
from pyxnat import Interface
import os.path as op
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_info_scan():
    assert scan_1.exists()
    expected_output = f'<Scan Object> 11 (`SPGR` 175 frames)  {central._server}/data/projects/pyxnat_tests/subjects/BBRCDEV_S02627/experiments/BBRCDEV_E03106/scans/11?format=html'
    assert str(scan_1) == expected_output