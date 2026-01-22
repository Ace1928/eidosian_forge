from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def is_state(self, state, model, allow_substates=False):
    if allow_substates:
        current = getattr(model, self.model_attribute)
        current_name = self.state_cls.separator.join(self._get_enum_path(current)) if isinstance(current, Enum) else current
        state_name = self.state_cls.separator.join(self._get_enum_path(state)) if isinstance(state, Enum) else state
        return current_name.startswith(state_name)
    return getattr(model, self.model_attribute) == state