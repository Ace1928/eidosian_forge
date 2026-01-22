import json
import re
import socket
import time
import zlib
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def fix_invalid_varnames(self, data):
    """Change ':'' and '-' to '_' to ensure valid template variable names"""
    new_data = data.copy()
    for key, value in data.items():
        if ':' in key or '-' in key:
            newkey = re.sub(':|-', '_', key)
            new_data[newkey] = value
            del new_data[key]
    return new_data