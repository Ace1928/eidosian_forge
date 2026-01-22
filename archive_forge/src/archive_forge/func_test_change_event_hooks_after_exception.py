import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
def test_change_event_hooks_after_exception(self):
    foo = Foo()

    def _on_foo_fuz_changed(obj, name, old, new):
        raise FuzException('function')
    foo.on_trait_change(_on_foo_fuz_changed, 'fuz')
    pre_tracer = self._collect_pre_notification_events
    post_tracer = self._collect_post_notification_events
    with trait_notifiers.change_event_tracers(pre_tracer, post_tracer):
        foo.fuz = 3
    self.assertEqual(len(self.pre_change_events), 4)
    self.assertEqual(len(self.post_change_events), 4)
    expected_pre_events = [(foo, 'fuz', 0.0, 3.0, foo._on_fuz_change_notification), (foo, 'bar', 0.0, 1.0, foo._bar_changed.__func__), (foo, 'bar', 0.0, 1.0, foo._on_bar_change_notification), (foo, 'fuz', 0.0, 3.0, _on_foo_fuz_changed)]
    self.assertEqual(self.pre_change_events, expected_pre_events)
    expected_post_events = [(foo, 'bar', 0.0, 1.0, foo._bar_changed.__func__), (foo, 'bar', 0.0, 1.0, foo._on_bar_change_notification), (foo, 'fuz', 0.0, 3.0, foo._on_fuz_change_notification), (foo, 'fuz', 0.0, 3.0, _on_foo_fuz_changed)]
    self.assertEqual(self.post_change_events, expected_post_events)
    self.assertEqual(self.exceptions[:2], [None, None])
    self.assertIsInstance(self.exceptions[2], FuzException)
    self.assertEqual(self.exceptions[2].args, ('method',))
    self.assertIsInstance(self.exceptions[3], FuzException)
    self.assertEqual(self.exceptions[3].args, ('function',))