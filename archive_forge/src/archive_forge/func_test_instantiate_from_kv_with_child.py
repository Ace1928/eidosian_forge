import unittest
import textwrap
from collections import defaultdict
def test_instantiate_from_kv_with_child(self):
    from kivy.lang import Builder

    class TestEventsFromKVChild(TrackCallbacks.get_base_class()):
        instantiated_widgets = []
    widget = Builder.load_string(textwrap.dedent("\n        TestEventsFromKVChild:\n            events_in_post: [1, 2]\n            on_kv_pre: self.add(2, 'pre')\n            on_kv_applied: self.add(2, 'applied')\n            on_kv_post: self.add(2, 'post')\n            root_widget: self\n            base_widget: self\n            name: 'root'\n            my_roots_expected_ids: {'child_widget': child_widget}\n            TestEventsFromKVChild:\n                events_in_post: [1, 2]\n                on_kv_pre: self.add(2, 'pre')\n                on_kv_applied: self.add(2, 'applied')\n                on_kv_post: self.add(2, 'post')\n                root_widget: root\n                base_widget: root\n                name: 'child'\n                id: child_widget\n                my_roots_expected_ids: {'child_widget': self}\n        "))
    self.assertIsInstance(widget, TestEventsFromKVChild)
    widget.check(self)