from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def vmware_argument_spec():
    return dict(hostname=dict(type='str', required=False, fallback=(env_fallback, ['VMWARE_HOST'])), username=dict(type='str', aliases=['user', 'admin'], required=False, fallback=(env_fallback, ['VMWARE_USER'])), password=dict(type='str', aliases=['pass', 'pwd'], required=False, no_log=True, fallback=(env_fallback, ['VMWARE_PASSWORD'])), port=dict(type='int', default=443, fallback=(env_fallback, ['VMWARE_PORT'])), validate_certs=dict(type='bool', required=False, default=True, fallback=(env_fallback, ['VMWARE_VALIDATE_CERTS'])), proxy_host=dict(type='str', required=False, default=None, fallback=(env_fallback, ['VMWARE_PROXY_HOST'])), proxy_port=dict(type='int', required=False, default=None, fallback=(env_fallback, ['VMWARE_PROXY_PORT'])))