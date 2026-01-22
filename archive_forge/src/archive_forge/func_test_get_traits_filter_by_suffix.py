import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_get_traits_filter_by_suffix(self):
    traits = ot.get_traits(suffix='SSE42')
    self.assertIn('HW_CPU_X86_SSE42', traits)
    self.assertEqual(1, len(traits))