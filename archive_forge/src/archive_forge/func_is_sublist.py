import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def is_sublist(source, target):
    """Checks if one list is a sublist of another.

    Arguments:
      source: the list in which to search for the occurrence of target.
      target: the list to search for as a sublist of source

    Returns:
      true if target is in source; false otherwise
    """
    for index in (i for i, e in enumerate(source) if e == target[0]):
        if tuple(source[index:index + len(target)]) == target:
            return True
    return False