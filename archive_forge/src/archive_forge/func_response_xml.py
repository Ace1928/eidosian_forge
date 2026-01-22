from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def response_xml(self, rawoutput):
    """Handle APIC XML response output"""
    try:
        xml = lxml.etree.fromstring(to_bytes(rawoutput))
        xmldata = cobra.data(xml)
    except Exception as e:
        self.error = dict(code=-1, text="Unable to parse output as XML, see 'raw' output. {0}".format(e))
        self.result['raw'] = rawoutput
        return
    self.imdata = xmldata.get('imdata', {}).get('children')
    if self.imdata is None:
        self.imdata = dict()
    self.totalCount = int(xmldata.get('imdata', {}).get('attributes', {}).get('totalCount', -1))
    self.response_error()