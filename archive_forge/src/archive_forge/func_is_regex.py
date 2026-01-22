import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts('regex', raises=re.error)
def is_regex(instance):
    if not isinstance(instance, str_types):
        return True
    return re.compile(instance)