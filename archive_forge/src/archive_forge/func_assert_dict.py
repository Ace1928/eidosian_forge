import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
def assert_dict(self, value, state):
    """
        Helper to assure we have proper input
        """
    if not hasattr(value, 'items'):
        raise Invalid(self.message('badDictType', state, type=type(value), value=value), value, state)