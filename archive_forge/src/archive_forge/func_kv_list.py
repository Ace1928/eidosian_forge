from __future__ import absolute_import, division, print_function
import json
import os
import re
import shutil
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible.module_utils.six import PY2, PY3, binary_type, iteritems, string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode, urlsplit
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.compat.datetime import utcnow, utcfromtimestamp
from ansible.module_utils.six.moves.collections_abc import Mapping, Sequence
from ansible.module_utils.urls import fetch_url, get_response_filename, parse_content_type, prepare_multipart, url_argument_spec
def kv_list(data):
    """ Convert data into a list of key-value tuples """
    if data is None:
        return None
    if isinstance(data, Sequence):
        return list(data)
    if isinstance(data, Mapping):
        return list(data.items())
    raise TypeError('cannot form-urlencode body, expect list or dict')