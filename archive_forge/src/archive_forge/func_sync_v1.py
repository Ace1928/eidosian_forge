from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def sync_v1(state, client, path, payload, check_mode, compare=do_differ_v1):
    changed, result = sync(state, client, path, payload, check_mode, compare)
    return (changed, convert_v1_to_v2_response(result))