from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_states(self, states):
    """ Retrieves a list of NestedStates.
        Args:
            states (str, Enum or list of str or Enum): Names/values of the states to retrieve.
        Returns:
            list(NestedStates) belonging to the passed identifiers.
        """
    res = []
    for state in states:
        if isinstance(state, list):
            res.append(self.get_states(state))
        else:
            res.append(self.get_state(state))
    return res