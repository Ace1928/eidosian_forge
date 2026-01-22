import sys
import time
from . import errors
from .helpers import newobject as object
def validate_parameters(self):
    """Validate that the parameters are correctly specified."""
    for p in self.params:
        if p not in self.known_params:
            raise errors.UnknownParameter(p, self.known_params)