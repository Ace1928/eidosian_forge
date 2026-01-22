import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def confirm_type(self, value, state):
    for t in self.type:
        if type(value) is t:
            break
    else:
        if len(self.type) == 1:
            msg = self.message('type', state, object=value, type=self.type[0])
        else:
            msg = self.message('inType', state, object=value, typeList=', '.join(map(str, self.type)))
        raise Invalid(msg, value, state)
    return value