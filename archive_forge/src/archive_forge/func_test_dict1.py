import unittest
from traits.api import (
def test_dict1(self):
    d1 = Dict1(tc=self)
    for i in range(3):
        ac = ArgCheckBase()
        d1.trait_set(exp_object=d1, exp_name='refs_items', type_old=None, exp_old=Undefined, type_new=TraitDictEvent)
        d1.refs[i] = ac
    self.assertEqual(d1.calls, {0: 3, 3: 0, 4: 0}, 'Behavior of a bug (#538) is not reproduced.')
    for i in range(3):
        self.assertEqual(d1.refs[i].value, 0)
    d1.reset_traits(['calls'])
    refs = {0: ArgCheckBase(), 1: ArgCheckBase(), 2: ArgCheckBase()}
    d1.trait_set(exp_object=d1, exp_name='refs', type_old=None, exp_old=d1.refs, type_new=TraitDictObject)
    d1.refs = refs
    self.assertEqual(d1.calls, {0: 1, 3: 1, 4: 1})
    for i in range(3):
        self.assertEqual(d1.refs[i].value, 0)
    d1.reset_traits(['calls'])
    for i in range(3):
        for j in range(3):
            d1.trait_set(exp_object=d1.refs[j], exp_name='value', type_old=None, exp_old=i, type_new=None, exp_new=i + 1)
            d1.refs[j].value = i + 1
    self.assertEqual(d1.calls, {0: 9, 3: 9, 4: 9})
    for i in range(3):
        self.assertEqual(d1.refs[i].value, 3)