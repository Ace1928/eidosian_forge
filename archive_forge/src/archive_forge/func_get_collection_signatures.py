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
@g_connect(['v2', 'v3'])
def get_collection_signatures(self, namespace, name, version):
    """
        Gets the collection signatures from the Galaxy server about a specific Collection version.

        :param namespace: The collection namespace.
        :param name: The collection name.
        :param version: Version of the collection to get the information for.
        :return: A list of signature strings.
        """
    api_path = self.available_api_versions.get('v3', self.available_api_versions.get('v2'))
    url_paths = [self.api_server, api_path, 'collections', namespace, name, 'versions', version, '/']
    n_collection_url = _urljoin(*url_paths)
    error_context_msg = 'Error when getting collection version metadata for %s.%s:%s from %s (%s)' % (namespace, name, version, self.name, self.api_server)
    data = self._call_galaxy(n_collection_url, error_context_msg=error_context_msg, cache=True)
    self._set_cache()
    signatures = [signature_info['signature'] for signature_info in data.get('signatures') or []]
    if not signatures:
        display.vvvv(f'Server {self.api_server} has not signed {namespace}.{name}:{version}')
    return signatures