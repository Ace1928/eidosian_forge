from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
@initial.setter
def initial(self, value):
    self._initial = self._recursive_initial(value)