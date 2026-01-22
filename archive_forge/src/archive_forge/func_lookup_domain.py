from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def lookup_domain(self, domain, ignore_not_found_error=False):
    """Look up a domain and return its id"""
    if domain is None:
        return domain
    d = self.get_obj('auth/domains', key='domains', name=domain)
    if not d and (not ignore_not_found_error):
        self.fail_json(msg="Domain '{0}' is not a valid domain name.".format(domain))
    elif (not d or 'id' not in d) and ignore_not_found_error:
        self.module.warn("Domain '{0}' is not a valid domain name.".format(domain))
        return None
    if 'id' not in d:
        self.fail_json(msg="Domain lookup failed for domain '{0}': {1}".format(domain, d))
    return d.get('id')