from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_global_name(self, state=None, join=True):
    """ Returns the name of the passed state in context of the current prefix/scope.
        Args:
            state (str, Enum or NestedState): The state to be analyzed.
            join (bool): Whether this method should join the path elements or not
        Returns:
            str or list(str) of the global state name
        """
    domains = copy.copy(self.prefix_path)
    if state:
        state_name = state.name if hasattr(state, 'name') else state
        if state_name in self.states:
            domains.append(state_name)
        else:
            raise ValueError("State '{0}' not found in local states.".format(state))
    return self.state_cls.separator.join(domains) if join else domains