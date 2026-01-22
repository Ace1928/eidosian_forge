import unittest
from traits.api import (
def test_arg_check_simple(self):
    ac = ArgCheckSimple(tc=self)
    ac.on_trait_change(ac.arg_check0, 'value')
    ac.on_trait_change(ac.arg_check1, 'value')
    ac.on_trait_change(ac.arg_check2, 'value')
    ac.on_trait_change(ac.arg_check3, 'value')
    ac.on_trait_change(ac.arg_check4, 'value')
    for i in range(3):
        ac.value += 1
    self.assertEqual(ac.calls, 3 * 5)
    ac.on_trait_change(ac.arg_check0, 'value', remove=True)
    ac.on_trait_change(ac.arg_check1, 'value', remove=True)
    ac.on_trait_change(ac.arg_check2, 'value', remove=True)
    ac.on_trait_change(ac.arg_check3, 'value', remove=True)
    ac.on_trait_change(ac.arg_check4, 'value', remove=True)
    for i in range(3):
        ac.value += 1
    self.assertEqual(ac.calls, 3 * 5)
    self.assertEqual(ac.value, 2 * 3)