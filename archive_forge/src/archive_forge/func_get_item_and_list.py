from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_item_and_list(rest_obj, name, uri, key='Name', value='value'):
    resp = rest_obj.invoke_request('GET', uri)
    tlist = []
    if resp.success and resp.json_data.get(value):
        tlist = resp.json_data.get(value, [])
        for xtype in tlist:
            if xtype.get(key, '') == name:
                return (xtype, tlist)
    return ({}, tlist)