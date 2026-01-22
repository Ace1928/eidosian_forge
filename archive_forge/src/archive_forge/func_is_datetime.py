import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts('date-time', raises=(ValueError, isodate.ISO8601Error))
def is_datetime(instance):
    if not isinstance(instance, str_types):
        return True
    return isodate.parse_datetime(instance)