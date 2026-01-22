from functools import update_wrapper
import numpy as np
def wrap_array_attribute(name):
    wrappee = getattr(np.ndarray, name)
    if wrappee is None:
        assert name == '__hash__'
        return None

    def attr(self):
        array = np.asarray(self)
        return getattr(array, name)
    update_wrapper(attr, wrappee)
    attr.__doc__ = None
    return property(attr)