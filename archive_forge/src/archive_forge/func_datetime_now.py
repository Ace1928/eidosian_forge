import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def datetime_now(module):
    if module.__name__ == 'datetime':
        return module.datetime.now()
    else:
        return module.now()