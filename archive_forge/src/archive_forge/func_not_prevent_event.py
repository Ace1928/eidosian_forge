import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def not_prevent_event(event):
    """ An implementation of prevent_event that does not prevent
    any event from being propagated.
    """
    return False