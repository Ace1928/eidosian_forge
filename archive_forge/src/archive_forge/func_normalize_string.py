from __future__ import absolute_import, division, print_function
import base64
import json
import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
def normalize_string(filedata):
    filedata = filedata.replace(':', '-')
    filedata = filedata.replace('.', '-')
    return filedata