from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def get_assume_role(params):
    """ Return new params """
    sts_params = get_acs_connection_info(params)
    assume_role = {}
    if params.get('assume_role'):
        assume_role['alicloud_assume_role_arn'] = params['assume_role'].get('role_arn')
        assume_role['alicloud_assume_role_session_name'] = params['assume_role'].get('session_name')
        assume_role['alicloud_assume_role_session_expiration'] = params['assume_role'].get('session_expiration')
        assume_role['alicloud_assume_role_policy'] = params['assume_role'].get('policy')
    assume_role_params = {'role_arn': params.get('alicloud_assume_role_arn') if params.get('alicloud_assume_role_arn') else assume_role.get('alicloud_assume_role_arn'), 'role_session_name': params.get('alicloud_assume_role_session_name') if params.get('alicloud_assume_role_session_name') else assume_role.get('alicloud_assume_role_session_name'), 'duration_seconds': params.get('alicloud_assume_role_session_expiration') if params.get('alicloud_assume_role_session_expiration') else assume_role.get('alicloud_assume_role_session_expiration', 3600), 'policy': assume_role.get('alicloud_assume_role_policy', {})}
    try:
        sts = connect_to_acs(footmark.sts, params.get('alicloud_region'), **sts_params).assume_role(**assume_role_params).read()
        sts_params['acs_access_key_id'], sts_params['acs_secret_access_key'], sts_params['security_token'] = (sts['access_key_id'], sts['access_key_secret'], sts['security_token'])
    except AnsibleACSError as e:
        params.fail_json(msg=str(e))
    return sts_params