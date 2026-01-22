import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
def get_item_type(self, model_name):
    try:
        model_type = check_call(self.collection.get_item_type, model_name)
    except KimpyError:
        msg = 'Could not find model {} installed in any of the KIM API model collections on this system.  See https://openkim.org/doc/usage/obtaining-models/ for instructions on installing models.'.format(model_name)
        raise KIMModelNotFound(msg)
    return model_type