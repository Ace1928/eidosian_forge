import functools
import threading
def threadsafe_method(func):
    """Marks a method of a ThreadSafeSingleton-derived class as inherently thread-safe.

    A method so marked must either not use any singleton state, or lock it appropriately.
    """
    func.is_threadsafe_method = True
    return func