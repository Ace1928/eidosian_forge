import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_get_traits_filter_by_prefix_and_suffix(self):
    traits = ot.get_traits(prefix='HW_NIC', suffix='RSA')
    self.assertIn('HW_NIC_ACCEL_RSA', traits)
    self.assertNotIn(ot.HW_NIC_ACCEL_TLS, traits)
    self.assertEqual(1, len(traits))
    traits = ot.get_traits(prefix='HW_NIC', suffix='TX')
    self.assertIn('HW_NIC_SRIOV_QOS_TX', traits)
    self.assertIn('HW_NIC_OFFLOAD_TX', traits)
    self.assertEqual(2, len(traits))