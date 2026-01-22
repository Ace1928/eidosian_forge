from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def remove_get_keys_from_payload_dict(payload_dict, remove_key_list):
    for each_key in remove_key_list:
        if each_key in payload_dict:
            payload_dict.pop(each_key)
    return payload_dict