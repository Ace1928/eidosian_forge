import unittest
import textwrap
from collections import defaultdict
def test_instantiate_from_kv_with_event(self):
    from kivy.lang import Builder

    class TestEventsFromKVEvent(TrackCallbacks.get_base_class()):
        instantiated_widgets = []
    widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVEvent:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n        "))
    self.assertIsInstance(widget, TestEventsFromKVEvent)
    widget.check(self)