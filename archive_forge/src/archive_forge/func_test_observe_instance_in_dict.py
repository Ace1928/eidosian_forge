import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_instance_in_dict(self):
    container = ClassWithDictOfInstance()
    events = []
    handler = events.append
    container.observe(handler=handler, expression=trait('name_to_instance', notify=False).dict_items(notify=False).trait('value'))
    single_value_instance = SingleValue()
    container.name_to_instance = {'name': single_value_instance}
    self.assertEqual(len(events), 0)
    single_value_instance.value += 1
    event, = events
    self.assertEqual(event.object, single_value_instance)
    self.assertEqual(event.name, 'value')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 1)