import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def validate_entry(entry, spec, val, missing, ret_true, ret_false):
    section.default_values.pop(entry, None)
    try:
        section.default_values[entry] = validator.get_default_value(configspec[entry])
    except (KeyError, AttributeError, validator.baseErrorClass):
        pass
    try:
        check = validator.check(spec, val, missing=missing)
    except validator.baseErrorClass as e:
        if not preserve_errors or isinstance(e, self._vdtMissingValue):
            out[entry] = False
        else:
            out[entry] = e
            ret_false = False
        ret_true = False
    else:
        ret_false = False
        out[entry] = True
        if self.stringify or missing:
            if not self.stringify:
                if isinstance(check, (list, tuple)):
                    check = [self._str(item) for item in check]
                elif missing and check is None:
                    check = ''
                else:
                    check = self._str(check)
            if check != val or missing:
                section[entry] = check
        if not copy and missing and (entry not in section.defaults):
            section.defaults.append(entry)
    return (ret_true, ret_false)