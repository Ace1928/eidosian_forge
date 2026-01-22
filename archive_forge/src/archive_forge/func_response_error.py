from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def response_error(self):
    """Set error information when found"""
    if self.totalCount != '0':
        try:
            self.error = self.imdata[0].get('error').get('attributes')
        except (AttributeError, IndexError, KeyError):
            pass