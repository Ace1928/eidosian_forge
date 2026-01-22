import unittest
from traits.adaptation.api import (
import traits.adaptation.tests.abc_examples
def test_global_register_offer(self):
    ex = self.examples
    offer = AdaptationOffer(factory=ex.UKStandardToEUStandard, from_protocol=ex.UKStandard, to_protocol=ex.EUStandard)
    register_offer(offer)
    uk_plug = ex.UKPlug()
    eu_plug = adapt(uk_plug, ex.EUStandard)
    self.assertIsNotNone(eu_plug)
    self.assertIsInstance(eu_plug, ex.UKStandardToEUStandard)