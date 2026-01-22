import timeit
import math
def timethis(name):

    def decorator(func):
        global _do_timings
        if name not in _do_timings:
            return func

        def wrapper(*args, **kwargs):
            from time import time
            global _timestack
            oldtimestack = _timestack
            _timestack = [func.func_name, [], 0, args]
            t1 = time()
            r = func(*args, **kwargs)
            t2 = time()
            _timestack[2] = t2 - t1
            if oldtimestack is not None:
                oldtimestack[1].append(_timestack)
                _timestack = oldtimestack
            else:
                _print_timestack(_timestack)
                _timestack = None
            return r
        return wrapper
    return decorator