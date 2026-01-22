import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def set_argument_pointer(self, compute_arg_name, data_object):
    return self.compute_args.set_argument_pointer(compute_arg_name, data_object)