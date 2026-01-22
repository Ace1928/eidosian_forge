import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_dunderinit_and_nondunderinit(self):
    """Make sure we can have both dunderinit'd traits and submodules
        co-exist in the same namespace.
        """
    traits = ot.get_traits('COMPUTE')
    self.assertIn('COMPUTE_DEVICE_TAGGING', traits)
    self.assertIn(ot.COMPUTE_DEVICE_TAGGING, traits)
    self.assertIn('COMPUTE_VOLUME_EXTEND', traits)
    self.assertIn(ot.COMPUTE_NET_ATTACH_INTERFACE, traits)