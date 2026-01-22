from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def ontap_sf_host_argument_spec():
    return dict(hostname=dict(required=True, type='str'), username=dict(required=True, type='str', aliases=['user']), password=dict(required=True, type='str', aliases=['pass'], no_log=True))