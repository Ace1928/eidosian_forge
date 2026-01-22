import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def get_neighbor_list_cutoffs_and_hints(self):
    return self.kim_model.get_neighbor_list_cutoffs_and_hints()