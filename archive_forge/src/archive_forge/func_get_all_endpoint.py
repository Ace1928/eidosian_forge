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
def get_all_endpoint(self, endpoint, *args, **kwargs):
    response = self.get_endpoint(endpoint, *args, **kwargs)
    if 'next' not in response['json']:
        raise RuntimeError('Expected list from API at {0}, got: {1}'.format(endpoint, response))
    next_page = response['json']['next']
    if response['json']['count'] > 10000:
        self.fail_json(msg='The number of items being queried for is higher than 10,000.')
    while next_page is not None:
        next_response = self.get_endpoint(next_page)
        response['json']['results'] = response['json']['results'] + next_response['json']['results']
        next_page = next_response['json']['next']
        response['json']['next'] = next_page
    return response