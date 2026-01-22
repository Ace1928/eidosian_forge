import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def datetime_isotime(module):
    if module.__name__ == 'datetime':
        return module.time.isoformat
    else:
        return module.ISO.Time