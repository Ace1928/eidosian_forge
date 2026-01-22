from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def is_valid_hex_color(color_choice):
    if re.match('^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color_choice):
        return True
    return False