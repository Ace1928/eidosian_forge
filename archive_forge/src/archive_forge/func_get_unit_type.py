from __future__ import absolute_import, division, print_function
import base64
import json
import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
def get_unit_type(filedata):
    beginsearch = filedata.find('unit_type_info="')
    beginsearch = beginsearch + 16
    endsearch = filedata.find('">', beginsearch)
    if (endsearch == -1) | (beginsearch == -1) | (endsearch < beginsearch) | (endsearch - beginsearch > 16):
        header = 'wti'
    else:
        header = filedata[beginsearch:beginsearch + (endsearch - beginsearch)]
    return header