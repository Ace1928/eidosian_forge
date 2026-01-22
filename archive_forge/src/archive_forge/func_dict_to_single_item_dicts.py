from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def dict_to_single_item_dicts(data):
    return [{k: v} for k, v in data.items()]