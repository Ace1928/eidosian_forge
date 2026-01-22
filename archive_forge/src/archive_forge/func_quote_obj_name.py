from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def quote_obj_name(object_name=None):
    """
    Replace special characters in object name
    with urllib quote equivalent

    """
    if not object_name:
        return None
    SPECIAL_CHARS = OrderedDict({'%': '%25', '/': '%2f', '\\': '%5c'})
    for key in SPECIAL_CHARS.keys():
        if key in object_name:
            object_name = object_name.replace(key, SPECIAL_CHARS[key])
    return object_name