import unittest
from traits.api import (
def test_pattern_list6(self):
    c = Complex(tc=self)
    c.on_trait_change(c.arg_check2, 'ref.[int1,int2,int3]')
    self.assertRaises(TraitError, c.trait_set, ref=ArgCheckBase())