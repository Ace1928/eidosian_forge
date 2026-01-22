from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_nested_triggers(self, src_path=None):
    """ Retrieves a list of valid triggers.
        Args:
            src_path (list(str)): A list representation of the source state's name.
        Returns:
            list(str) of valid trigger names.
        """
    if src_path:
        triggers = super(HierarchicalMachine, self).get_triggers(self.state_cls.separator.join(src_path))
        if len(src_path) > 1 and src_path[0] in self.states:
            with self(src_path[0]):
                triggers.extend(self.get_nested_triggers(src_path[1:]))
    else:
        triggers = list(self.events.keys())
        for state_name in self.states:
            with self(state_name):
                triggers.extend(self.get_nested_triggers())
    return triggers