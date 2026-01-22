from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def ecs_argument_spec():
    spec = acs_common_argument_spec()
    spec.update(dict(alicloud_region=dict(required=True, aliases=['region', 'region_id'], fallback=(env_fallback, ['ALICLOUD_REGION', 'ALICLOUD_REGION_ID'])), alicloud_assume_role_arn=dict(fallback=(env_fallback, ['ALICLOUD_ASSUME_ROLE_ARN']), aliases=['assume_role_arn']), alicloud_assume_role_session_name=dict(fallback=(env_fallback, ['ALICLOUD_ASSUME_ROLE_SESSION_NAME']), aliases=['assume_role_session_name']), alicloud_assume_role_session_expiration=dict(type='int', fallback=(env_fallback, ['ALICLOUD_ASSUME_ROLE_SESSION_EXPIRATION']), aliases=['assume_role_session_expiration']), alicloud_assume_role=dict(type='dict', aliases=['assume_role']), profile=dict(fallback=(env_fallback, ['ALICLOUD_PROFILE'])), shared_credentials_file=dict(fallback=(env_fallback, ['ALICLOUD_SHARED_CREDENTIALS_FILE']))))
    return spec