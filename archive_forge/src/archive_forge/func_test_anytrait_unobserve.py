import unittest
from traits.api import (
from traits.observation.api import (
def test_anytrait_unobserve(self):
    obj = HasVariousTraits()
    events = []
    obj.observe(events.append, '*')
    obj.foo = 23
    obj.bar = 'on'
    self.assertEqual(len(events), 2)
    obj.observe(events.append, '*', remove=True)
    obj.foo = 232
    obj.bar = 'mid'
    self.assertEqual(len(events), 2)