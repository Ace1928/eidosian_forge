import functools
import warnings
import threading
import sys
def check_returns(f):

    def new_f(*args, **kwds):
        result = f(*args, **kwds)
        assert isinstance(result, rtype), 'return value %r does not match %s' % (result, rtype)
        return result
    new_f.__name__ = f.__name__
    return new_f