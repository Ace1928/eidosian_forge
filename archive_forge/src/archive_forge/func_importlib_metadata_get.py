import collections
from importlib import util
import inspect
import sys
def importlib_metadata_get(group):
    ep = importlib_metadata.entry_points()
    if hasattr(ep, 'select'):
        return ep.select(group=group)
    else:
        return ep.get(group, ())