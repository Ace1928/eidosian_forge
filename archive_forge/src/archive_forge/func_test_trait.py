import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_trait(self):
    """Simply tests that the constants from submodules are imported into
        the primary os_traits module space.
        """
    trait = ot.HW_CPU_X86_SSE42
    self.assertEqual('HW_CPU_X86_SSE42', trait)
    self.assertEqual(x86.SSE42, ot.HW_CPU_X86_SSE42)
    self.assertEqual(api.DIRECTX_V10, ot.HW_GPU_API_DIRECTX_V10)
    self.assertEqual(resolution.W1920H1080, ot.HW_GPU_RESOLUTION_W1920H1080)
    self.assertEqual(offload.TSO, ot.HW_NIC_OFFLOAD_TSO)