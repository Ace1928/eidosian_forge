import weakref
from pydispatch import saferef, robustapply, errors
def getReceivers(sender=Any, signal=Any):
    """Get list of receivers from global tables

    This utility function allows you to retrieve the
    raw list of receivers from the connections table
    for the given sender and signal pair.

    Note:
        there is no guarantee that this is the actual list
        stored in the connections table, so the value
        should be treated as a simple iterable/truth value
        rather than, for instance a list to which you
        might append new records.

    Normally you would use liveReceivers( getReceivers( ...))
    to retrieve the actual receiver objects as an iterable
    object.
    """
    try:
        return connections[id(sender)][signal]
    except KeyError:
        return []