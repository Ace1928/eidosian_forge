import unittest
import textwrap
from collections import defaultdict
@staticmethod
def get_base_class():
    """The base class to use for widgets during testing so we can use
        this class variables to ease testing.
        """
    from kivy.uix.widget import Widget

    class TestEventsBase(TrackCallbacks, Widget):
        __events__ = ('on_kv_pre', 'on_kv_applied')
        instantiated_widgets = []
        events_in_pre = [1]
        events_in_applied = [1]
        events_in_post = [1]

        def on_kv_pre(self):
            self.add(1, 'pre')

        def on_kv_applied(self, root_widget):
            self.add(1, 'applied')
            self.actual_root_widget = root_widget

        def on_kv_post(self, base_widget):
            self.add(1, 'post')
            self.actual_base_widget = base_widget
            self.actual_prop_values = {k: getattr(self, k) for k in self.expected_prop_values}
            if self.actual_root_widget is not None:
                self.actual_ids = dict(self.actual_root_widget.ids)

        def apply_class_lang_rules(self, root=None, **kwargs):
            self.dispatch('on_kv_pre')
            super(TestEventsBase, self).apply_class_lang_rules(root=root, **kwargs)
            self.dispatch('on_kv_applied', root)
    return TestEventsBase