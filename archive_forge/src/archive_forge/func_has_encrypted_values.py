from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
@staticmethod
def has_encrypted_values(obj):
    """Returns True if JSON-like python content in obj has $encrypted$
        anywhere in the data as a value
        """
    if isinstance(obj, dict):
        for val in obj.values():
            if ControllerAPIModule.has_encrypted_values(val):
                return True
    elif isinstance(obj, list):
        for val in obj:
            if ControllerAPIModule.has_encrypted_values(val):
                return True
    elif obj == ControllerAPIModule.ENCRYPTED_STRING:
        return True
    return False