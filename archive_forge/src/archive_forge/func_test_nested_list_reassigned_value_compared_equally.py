import unittest
from traits.api import (
from traits.observation.api import (
def test_nested_list_reassigned_value_compared_equally(self):
    container = ClassWithListOfListOfInstance()
    events = []
    handler = events.append
    container.observe(expression=trait('list_of_list_of_instances', notify=False).list_items(notify=False).list_items(notify=False).trait('value'), handler=handler)
    inner_list = [SingleValue()]
    container.list_of_list_of_instances = [inner_list]
    self.assertEqual(len(events), 0)
    container.list_of_list_of_instances[0] = inner_list
    second_instance = SingleValue()
    container.list_of_list_of_instances[0].append(second_instance)
    self.assertEqual(len(events), 0)
    second_instance.value += 1
    event, = events
    self.assertEqual(event.object, second_instance)
    self.assertEqual(event.name, 'value')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 1)