import weakref
from pydispatch import saferef, robustapply, errors
def liveReceivers(receivers):
    """Filter sequence of receivers to get resolved, live receivers

    This is a generator which will iterate over
    the passed sequence, checking for weak references
    and resolving them, then returning all live
    receivers.
    """
    for receiver in receivers:
        if isinstance(receiver, WEAKREF_TYPES):
            receiver = receiver()
            if receiver is not None:
                yield receiver
        else:
            yield receiver