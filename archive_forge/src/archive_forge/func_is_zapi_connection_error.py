from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def is_zapi_connection_error(message):
    """ return True if it is a connection issue """
    try:
        if isinstance(message, tuple) and isinstance(message[0], ConnectionError):
            return True
    except NameError:
        pass
    return isinstance(message, str) and message.startswith(('URLError', 'Unauthorized'))