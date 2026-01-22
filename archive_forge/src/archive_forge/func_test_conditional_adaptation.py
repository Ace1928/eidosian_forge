import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_conditional_adaptation(self):
    ex = self.examples

    def travel_plug_to_eu_standard(adaptee):
        if adaptee.mode == 'Europe':
            return ex.TravelPlugToEUStandard(adaptee=adaptee)
        else:
            return None
    self.adaptation_manager.register_factory(factory=travel_plug_to_eu_standard, from_protocol=ex.TravelPlug, to_protocol=ex.EUStandard)
    travel_plug = ex.TravelPlug(mode='Europe')
    eu_plug = self.adaptation_manager.adapt(travel_plug, ex.EUStandard)
    self.assertIsNotNone(eu_plug)
    self.assertIsInstance(eu_plug, ex.TravelPlugToEUStandard)
    travel_plug = ex.TravelPlug(mode='Asia')
    eu_plug = self.adaptation_manager.adapt(travel_plug, ex.EUStandard, None)
    self.assertIsNone(eu_plug)