from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
@staticmethod
def ensure_scaleway_secret_package(module):
    if not HAS_SCALEWAY_SECRET_PACKAGE:
        module.fail_json(msg=missing_required_lib('passlib[argon2]', url='https://passlib.readthedocs.io/en/stable/'), exception=SCALEWAY_SECRET_IMP_ERR)