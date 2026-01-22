from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def on_enter(self, state_name, callback):
    """ Helper function to add callbacks to states in case a custom state separator is used.
        Args:
            state_name (str): Name of the state
            callback (str or callable): Function to be called. Strings will be resolved to model functions.
        """
    self.get_state(state_name).add_callback('enter', callback)