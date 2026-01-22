from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import json
import ansible.module_utils.six.moves.urllib.error as urllib_error
def check_zone(data, name):
    """
    Returns true if zone already exists, and false if not.
    """
    counter = 0
    exists = False
    if data.status_code in [201, 200]:
        for zone in data.json():
            if zone['nickname'] == name:
                counter += 1
        if counter == 1:
            exists = True
    return (exists, counter)