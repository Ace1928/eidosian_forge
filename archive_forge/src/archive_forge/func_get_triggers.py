from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_triggers(self, *args):
    """ Extends transitions.core.Machine.get_triggers to also include parent state triggers. """
    triggers = []
    with self():
        for state in args:
            state_name = state.name if hasattr(state, 'name') else state
            state_path = state_name.split(self.state_cls.separator)
            if len(state_path) > 1:
                with self(state_path[0]):
                    triggers.extend(self.get_nested_triggers(state_path[1:]))
            while state_path:
                triggers.extend(super(HierarchicalMachine, self).get_triggers(self.state_cls.separator.join(state_path)))
                state_path.pop()
    return triggers