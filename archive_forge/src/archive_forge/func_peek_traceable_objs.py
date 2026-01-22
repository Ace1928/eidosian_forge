import inspect
def peek_traceable_objs(self):
    """Return iterator over stored TraceableObjects ordered newest to oldest."""
    return reversed(self._stack)