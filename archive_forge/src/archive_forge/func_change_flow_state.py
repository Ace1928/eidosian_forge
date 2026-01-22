import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
@fasteners.write_locked
def change_flow_state(self, state):
    """Transition flow from old state to new state.

        Returns ``(True, old_state)`` if transition was performed,
        or ``(False, old_state)`` if it was ignored, or raises a
        :py:class:`~taskflow.exceptions.InvalidState` exception if transition
        is invalid.
        """
    old_state = self.get_flow_state()
    if not states.check_flow_transition(old_state, state):
        return (False, old_state)
    self.set_flow_state(state)
    return (True, old_state)