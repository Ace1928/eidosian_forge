import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts('email')
def is_email(instance):
    if not isinstance(instance, str_types):
        return True
    return '@' in instance