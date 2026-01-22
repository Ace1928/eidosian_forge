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
def get_item_name(self, item, allow_unknown=False):
    if item:
        if 'name' in item:
            return item['name']
        for field_name in ControllerAPIModule.IDENTITY_FIELDS.values():
            if field_name in item:
                return item[field_name]
        if item.get('type', None) in ('o_auth2_access_token', 'credential_input_source'):
            return item['id']
    if allow_unknown:
        return 'unknown'
    if item:
        self.exit_json(msg='Cannot determine identity field for {0} object.'.format(item.get('type', 'unknown')))
    else:
        self.exit_json(msg='Cannot determine identity field for Undefined object.')