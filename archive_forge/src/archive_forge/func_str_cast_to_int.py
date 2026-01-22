import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def str_cast_to_int(object, name, value):
    """ A function that validates the value is a str and then converts
    it to an int using its length.
    """
    if not isinstance(value, str):
        raise TraitError('Not a string!')
    return len(value)