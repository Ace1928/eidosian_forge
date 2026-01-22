import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def get_species_support_and_code(self, species_name):
    return self.kim_model.get_species_support_and_code(species_name)