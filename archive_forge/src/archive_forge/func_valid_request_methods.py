import functools
import logging
from ..errors import (
@valid_request_methods.setter
def valid_request_methods(self, valid_request_methods):
    if valid_request_methods is not None:
        valid_request_methods = [x.upper() for x in valid_request_methods]
    self._valid_request_methods = valid_request_methods