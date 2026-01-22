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
def lookup_users(self, users, ignore_not_found_error=False):
    """Look up users and return their ids"""
    if users is None:
        users = ['admin']
    elif 'admin' not in users:
        users.append('admin')
    ids = []
    for user in users:
        if self.platform == 'nd':
            u = self.get_obj('users', loginID=user, api_version='v2')
        else:
            u = self.get_obj('users', username=user)
        if not u and (not ignore_not_found_error):
            self.fail_json(msg="User '{0}' is not a valid user name.".format(user))
        elif (not u or 'id' not in u) and ignore_not_found_error:
            self.module.warn("User '{0}' is not a valid user name.".format(user))
            return ids
        if 'id' not in u:
            if 'userID' not in u:
                self.fail_json(msg="User lookup failed for user '{0}': {1}".format(user, u))
            id = dict(userId=u.get('userID'))
        else:
            id = dict(userId=u.get('id'))
        if id in ids:
            self.fail_json(msg="User '{0}' is duplicate.".format(user))
        ids.append(id)
    return ids