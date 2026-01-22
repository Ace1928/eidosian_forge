from functools import wraps
def registerAsIOThread():
    """Mark the current thread as responsible for I/O requests."""
    global ioThread
    ioThread = getThreadID()