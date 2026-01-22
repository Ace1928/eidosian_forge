import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_no_adapter_available(self):
    ex = self.examples
    plug = ex.UKPlug()
    eu_plug = self.adaptation_manager.adapt(plug, ex.EUPlug, None)
    self.assertEqual(eu_plug, None)
    eu_plug = self.adaptation_manager.adapt(plug, ex.EUStandard, None)
    self.assertEqual(eu_plug, None)