import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
@unittest.skipIf(not QT_FOUND, 'Qt event loop not found, UI dispatch not possible.')
def test_notification_from_main_thread(self):
    obj = self.obj_factory()
    obj.foo = 3
    self.flush_event_loop()
    notifications = self.notifications
    self.assertEqual(len(notifications), 1)
    thread_id, event = notifications[0]
    self.assertEqual(event, (obj, 'foo', 0, 3))
    ui_thread = trait_notifiers.ui_thread
    self.assertEqual(thread_id, ui_thread)