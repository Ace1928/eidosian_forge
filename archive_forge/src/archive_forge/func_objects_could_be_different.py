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
def objects_could_be_different(self, old, new, field_set=None, warning=False):
    if field_set is None:
        field_set = set((fd for fd in new.keys() if fd not in ('modified', 'related', 'summary_fields')))
    for field in field_set:
        new_field = new.get(field, None)
        old_field = old.get(field, None)
        if old_field != new_field:
            if self.update_secrets or not self.fields_could_be_same(old_field, new_field):
                return True
        elif self.has_encrypted_values(new_field) or field not in new:
            if self.update_secrets or not self.fields_could_be_same(old_field, new_field):
                self._encrypted_changed_warning(field, old, warning=warning)
                return True
    return False