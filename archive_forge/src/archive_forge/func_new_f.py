import functools
import warnings
import threading
import sys
def new_f(*args, **kwds):
    result = f(*args, **kwds)
    assert isinstance(result, rtype), 'return value %r does not match %s' % (result, rtype)
    return result