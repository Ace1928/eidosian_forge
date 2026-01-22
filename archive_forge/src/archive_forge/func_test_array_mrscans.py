import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log
@skip_if_no_network
def test_array_mrscans(self):
    """
        Get a list of MRI scans from a given experiment which has multiple
        scans mixed (i.e. MRScans and MRSpectroscopies, aka OtherDicomScans)
        and assert its length matches the list of scans filtered by type
        'xnat:mrScanData'
        """
    mris = self._intf.array.mrscans(experiment_id='NITRC_IR_E10539').data
    exps = self._intf.array.scans(experiment_id='NITRC_IR_E10539', scan_type='xnat:mrScanData').data
    self.assertListEqual([i['xnat:mrscandata/id'] for i in mris], [i['xnat:mrscandata/id'] for i in exps])