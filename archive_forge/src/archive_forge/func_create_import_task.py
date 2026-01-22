from __future__ import (absolute_import, division, print_function)
import collections
import datetime
import functools
import hashlib
import json
import os
import stat
import tarfile
import time
import threading
from http import HTTPStatus
from http.client import BadStatusLine, IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import quote as urlquote, urlencode, urlparse, parse_qs, urljoin
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.urls import open_url, prepare_multipart
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash_s
from ansible.utils.path import makedirs_safe
@g_connect(['v1'])
def create_import_task(self, github_user, github_repo, reference=None, role_name=None):
    """
        Post an import request
        """
    url = _urljoin(self.api_server, self.available_api_versions['v1'], 'imports') + '/'
    args = {'github_user': github_user, 'github_repo': github_repo, 'github_reference': reference if reference else ''}
    if role_name:
        args['alternate_role_name'] = role_name
    data = self._call_galaxy(url, args=urlencode(args), method='POST')
    if data.get('results', None):
        return data['results']
    return data