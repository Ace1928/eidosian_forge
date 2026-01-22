import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def datetime_makedate(module, year, month, day):
    if module.__name__ == 'datetime':
        return module.date(year, month, day)
    else:
        try:
            return module.DateTime(year, month, day)
        except module.RangeError as e:
            raise ValueError(str(e))