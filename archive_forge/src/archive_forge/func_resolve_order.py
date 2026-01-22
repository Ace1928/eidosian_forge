from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def resolve_order(state_tree):
    """ Converts a (model) state tree into a list of state paths. States are ordered in the way in which states
    should be visited to process the event correctly (Breadth-first). This makes sure that ALL children are evaluated
    before parents in parallel states.
    Args:
        state_tree (dict): A dictionary representation of the model's state.
    Returns:
        list of lists of str representing the order of states to be processed.
    """
    queue = []
    res = []
    prefix = []
    while True:
        for state_name in reversed(list(state_tree.keys())):
            scope = prefix + [state_name]
            res.append(scope)
            if state_tree[state_name]:
                queue.append((scope, state_tree[state_name]))
        if not queue:
            break
        prefix, state_tree = queue.pop(0)
    return reversed(res)