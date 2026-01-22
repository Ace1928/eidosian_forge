import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def set_callback_pointer(self, compute_callback_name, callback, data_object):
    return self.compute_args.set_callback_pointer(compute_callback_name, callback, data_object)