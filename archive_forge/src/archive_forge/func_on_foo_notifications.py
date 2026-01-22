import threading
import unittest
from unittest import mock
from traits.api import Float, HasTraits, List
from traits.testing.unittest_tools import UnittestTools
def on_foo_notifications(obj, name, old, new):
    thread_id = threading.current_thread().ident
    event = (thread_id, obj, name, old, new)
    receiver.notifications.append(event)