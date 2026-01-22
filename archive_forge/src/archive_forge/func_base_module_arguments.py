from __future__ import annotations
import traceback
from typing import Any, NoReturn
from ansible.module_utils.basic import AnsibleModule as AnsibleModuleBase, env_fallback
from ansible.module_utils.common.text.converters import to_native
from .client import ClientException, client_check_required_lib, client_get_by_name_or_id
from .vendor.hcloud import APIException, Client, HCloudException
from .vendor.hcloud.actions import ActionException
from .version import version
@classmethod
def base_module_arguments(cls):
    return {'api_token': {'type': 'str', 'required': True, 'fallback': (env_fallback, ['HCLOUD_TOKEN']), 'no_log': True}, 'api_endpoint': {'type': 'str', 'fallback': (env_fallback, ['HCLOUD_ENDPOINT']), 'default': 'https://api.hetzner.cloud/v1', 'aliases': ['endpoint']}}