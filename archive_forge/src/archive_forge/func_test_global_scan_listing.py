from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_global_scan_listing():
    assert central.array.scans(project_id='ixi', experiment_type='xnat:mrSessionData', scan_type='xnat:mrScanData')