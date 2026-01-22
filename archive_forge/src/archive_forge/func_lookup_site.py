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
def lookup_site(self, site, ignore_not_found_error=False):
    """Look up a site and return its id"""
    if site is None:
        return site
    s = self.get_obj('sites', name=site)
    if not s and (not ignore_not_found_error):
        self.fail_json(msg="Site '{0}' is not a valid site name.".format(site))
    elif (not s or 'id' not in s) and ignore_not_found_error:
        self.module.warn("Site '{0}' is not a valid site name.".format(site))
        return None
    if 'id' not in s:
        self.fail_json(msg="Site lookup failed for site '{0}': {1}".format(site, s))
    return s.get('id')