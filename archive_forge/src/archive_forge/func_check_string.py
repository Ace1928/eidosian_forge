from __future__ import absolute_import
import types
from . import Errors
def check_string(self, num, value):
    if type(value) != type(''):
        self.wrong_type(num, value, 'string')