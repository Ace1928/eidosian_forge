from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ..module_utils import arguments, errors, utils
def update_password_hash(client, path, username, password_hash, check_mode):
    if client.version < '5.21.0':
        raise errors.SensuError('Sensu Go < 5.21.0 does not support password hashes')
    if not check_mode:
        utils.put(client, path + '/reset_password', dict(username=username, password_hash=password_hash))
    return True