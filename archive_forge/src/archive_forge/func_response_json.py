from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def response_json(self, rawoutput):
    """Handle APIC JSON response output"""
    try:
        jsondata = json.loads(rawoutput)
    except Exception as e:
        self.error = dict(code=-1, text="Unable to parse output as JSON, see 'raw' output. {0}".format(e))
        self.result['raw'] = rawoutput
        return
    self.imdata = jsondata.get('imdata')
    if self.imdata is None:
        self.imdata = dict()
    self.totalCount = int(jsondata.get('totalCount'))
    self.response_error()