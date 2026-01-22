from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def to_state(self, model, state_name, *args, **kwargs):
    """ Helper function to add go to states in case a custom state separator is used.
        Args:
            model (class): The model that should be used.
            state_name (str): Name of the destination state.
        """
    current_state = getattr(model, self.model_attribute)
    if isinstance(current_state, list):
        raise MachineError("Cannot use 'to_state' from parallel state")
    event = NestedEventData(self.get_state(current_state), Event('to', self), self, model, args=args, kwargs=kwargs)
    if isinstance(current_state, Enum):
        event.source_path = self._get_enum_path(current_state)
        event.source_name = self.state_cls.separator.join(event.source_path)
    else:
        event.source_name = current_state
        event.source_path = current_state.split(self.state_cls.separator)
    self._create_transition(event.source_name, state_name).execute(event)