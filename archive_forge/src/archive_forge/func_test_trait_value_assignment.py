import unittest
from traits.api import Int, Float, String, HasRequiredTraits, TraitError
def test_trait_value_assignment(self):
    test_instance = RequiredTest(i_trait=4, f_trait=2.2, s_trait='test')
    self.assertEqual(test_instance.i_trait, 4)
    self.assertEqual(test_instance.f_trait, 2.2)
    self.assertEqual(test_instance.s_trait, 'test')
    self.assertEqual(test_instance.non_req_trait, 4.4)
    self.assertEqual(test_instance.normal_trait, 42.0)