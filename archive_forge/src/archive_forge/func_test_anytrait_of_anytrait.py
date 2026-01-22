import unittest
from traits.api import (
from traits.observation.api import (
def test_anytrait_of_anytrait(self):
    foo = HasVariousTraits()
    bar = HasVariousTraits()
    obj = UpdateListener(foo=foo, bar=bar)
    events = []
    with self.assertRaises(ValueError):
        obj.observe(events.append, '*:*')