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
def filter_sensitive_attributes(container, attributes):
    """
    WARNING: This function is effectively private, **do not use it**!
    It will be removed or renamed once changing its name no longer triggers a pylint bug.
    """
    for attr in attributes:
        container[attr] = 'SENSITIVE_VALUE'
    return container