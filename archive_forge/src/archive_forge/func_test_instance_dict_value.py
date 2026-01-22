import unittest
from traits.api import (
def test_instance_dict_value(self):
    inst = InstanceDictValue(tc=self)
    inst.trait_set(exp_object=inst.ref, exp_name='value', dst_name='value', exp_old={0: 0, 1: 1, 2: 2}, exp_new={0: 0, 1: 1, 2: 2, 3: 3}, dst_new={0: 0, 1: 1, 2: 2, 3: 3})
    inst.ref.value = {0: 0, 1: 1, 2: 2, 3: 3}
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, {0: 0, 1: 1, 2: 2, 3: 3})
    inst.reset_traits(['calls'])
    ref = ArgCheckDict()
    inst.trait_set(exp_object=inst, exp_name='ref', dst_name='value', exp_old=inst.ref, exp_new=ref, dst_new={0: 0, 1: 1, 2: 2})
    inst.ref = ref
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, {0: 0, 1: 1, 2: 2})
    inst.reset_traits(['calls'])
    inst.trait_set(exp_object=inst.ref, exp_name='value', dst_name='value', exp_old={0: 0, 1: 1, 2: 2}, exp_new={0: 0, 1: 1, 2: 2, 3: 3}, dst_new={0: 0, 1: 1, 2: 2, 3: 3})
    with self.assertRaises(AssertionError, msg='Behavior of a bug (#537) is not reproduced.'):
        inst.ref.value[3] = 3
    with self.assertRaises(AssertionError, msg='Behavior of a bug (#537) is not reproduced.'):
        self.assertEqual(inst.calls, {0: 1, 1: 0, 2: 0, 3: 0, 4: 0})
    self.assertEqual(inst.ref.value, {0: 0, 1: 1, 2: 2, 3: 3})