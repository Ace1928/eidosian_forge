import abc
import collections
import threading
from automaton import exceptions as machine_excp
from automaton import machines
import fasteners
import futurist
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.types import failure as ft
from taskflow.utils import schema_utils as su
def transition_and_log_error(self, new_state, logger=None):
    """Transitions *and* logs an error if that transitioning raises.

        This overlays the transition function and performs nearly the same
        functionality but instead of raising if the transition was not valid
        it logs a warning to the provided logger and returns False to
        indicate that the transition was not performed (note that this
        is *different* from the transition function where False means
        ignored).
        """
    if logger is None:
        logger = LOG
    moved = False
    try:
        moved = self.transition(new_state)
    except excp.InvalidState:
        logger.warn("Failed to transition '%s' to %s state.", self, new_state, exc_info=True)
    return moved