import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def mutEphemeral(individual, mode):
    """This operator works on the constants of the tree *individual*. In
    *mode* ``"one"``, it will change the value of one of the individual
    ephemeral constants by calling its generator function. In *mode*
    ``"all"``, it will change the value of **all** the ephemeral constants.

    :param individual: The normal or typed tree to be mutated.
    :param mode: A string to indicate to change ``"one"`` or ``"all"``
                 ephemeral constants.
    :returns: A tuple of one tree.
    """
    if mode not in ['one', 'all']:
        raise ValueError('Mode must be one of "one" or "all"')
    ephemerals_idx = [index for index, node in enumerate(individual) if isinstance(type(node), MetaEphemeral)]
    if len(ephemerals_idx) > 0:
        if mode == 'one':
            ephemerals_idx = (random.choice(ephemerals_idx),)
        for i in ephemerals_idx:
            individual[i] = type(individual[i])()
    return (individual,)