import functools
import threading
def autolocked_method(func):
    """Automatically synchronizes all calls of a method of a ThreadSafeSingleton-derived
    class by locking the singleton for the duration of each call.
    """

    @functools.wraps(func)
    @threadsafe_method
    def lock_and_call(self, *args, **kwargs):
        with self:
            return func(self, *args, **kwargs)
    return lock_and_call