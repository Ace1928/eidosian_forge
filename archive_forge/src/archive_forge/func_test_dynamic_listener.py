import unittest
from traits.api import HasTraits, Dict
def test_dynamic_listener(self):
    foo = MyOtherClass()
    func = Callback(self, added={'g': 'guava'})
    foo.on_trait_change(func.__call__, 'd_items')
    foo.d['g'] = 'guava'
    foo.on_trait_change(func.__call__, 'd_items', remove=True)
    self.assertTrue(func.called)
    func2 = Callback(self, removed={'a': 'apple'})
    foo.on_trait_change(func2.__call__, 'd_items')
    del foo.d['a']
    foo.on_trait_change(func2.__call__, 'd_items', remove=True)
    self.assertTrue(func2.called)
    func3 = Callback(self, changed={'b': 'banana'})
    foo.on_trait_change(func3.__call__, 'd_items')
    foo.d['b'] = 'broccoli'
    foo.on_trait_change(func3.__call__, 'd_items', remove=True)
    self.assertTrue(func3.called)