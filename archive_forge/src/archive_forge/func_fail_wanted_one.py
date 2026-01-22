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
def fail_wanted_one(self, response, endpoint, query_params):
    sample = response.copy()
    if len(sample['json']['results']) > 1:
        sample['json']['results'] = sample['json']['results'][:2] + ['...more results snipped...']
    url = self.build_url(endpoint, query_params)
    host_length = len(self.host)
    display_endpoint = url.geturl()[host_length:]
    self.fail_json(msg='Request to {0} returned {1} items, expected 1'.format(display_endpoint, response['json']['count']), query=query_params, response=sample, total_results=response['json']['count'])