import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def validate_partial(self, field_dict, state):
    if not field_dict.get(self.cc_type_field, None) or not field_dict.get(self.cc_code_field, None):
        return None
    self._validate_python(field_dict, state)