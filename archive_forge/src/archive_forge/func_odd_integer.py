import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def odd_integer(object, name, value):
    try:
        float(value)
        if value % 2 == 1:
            return int(value)
    except:
        pass
    raise TraitError