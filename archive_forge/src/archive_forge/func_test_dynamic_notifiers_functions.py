import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_dynamic_notifiers_functions(self):
    calls_0 = []

    def function_listener_0():
        calls_0.append(())
    calls_1 = []

    def function_listener_1(new):
        calls_1.append((new,))
    calls_2 = []

    def function_listener_2(name, new):
        calls_2.append((name, new))
    calls_3 = []

    def function_listener_3(obj, name, new):
        calls_3.append((obj, name, new))
    calls_4 = []

    def function_listener_4(obj, name, old, new):
        calls_4.append((obj, name, old, new))
    obj = DynamicNotifiers()
    obj.on_trait_change(function_listener_0, 'ok')
    obj.on_trait_change(function_listener_1, 'ok')
    obj.on_trait_change(function_listener_2, 'ok')
    obj.on_trait_change(function_listener_3, 'ok')
    obj.on_trait_change(function_listener_4, 'ok')
    obj.ok = 2
    obj.ok = 3
    expected_0 = [(), ()]
    self.assertEqual(expected_0, calls_0)
    expected_1 = [(2.0,), (3.0,)]
    self.assertEqual(expected_1, calls_1)
    expected_2 = [('ok', 2.0), ('ok', 3.0)]
    self.assertEqual(expected_2, calls_2)
    expected_3 = [(obj, 'ok', 2.0), (obj, 'ok', 3.0)]
    self.assertEqual(expected_3, calls_3)
    expected_4 = [(obj, 'ok', 0.0, 2.0), (obj, 'ok', 2.0, 3.0)]
    self.assertEqual(expected_4, calls_4)