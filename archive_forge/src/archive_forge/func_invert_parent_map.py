import time
from . import debug, errors, osutils, revision, trace
def invert_parent_map(parent_map):
    """Given a map from child => parents, create a map of parent=>children"""
    child_map = {}
    for child, parents in parent_map.items():
        for p in parents:
            if p not in child_map:
                child_map[p] = (child,)
            else:
                child_map[p] = child_map[p] + (child,)
    return child_map