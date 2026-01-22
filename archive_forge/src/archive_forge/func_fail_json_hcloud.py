from __future__ import annotations
import traceback
from typing import Any, NoReturn
from ansible.module_utils.basic import AnsibleModule as AnsibleModuleBase, env_fallback
from ansible.module_utils.common.text.converters import to_native
from .client import ClientException, client_check_required_lib, client_get_by_name_or_id
from .vendor.hcloud import APIException, Client, HCloudException
from .vendor.hcloud.actions import ActionException
from .version import version
def fail_json_hcloud(self, exception: HCloudException, msg: str | None=None, params: Any=None, **kwargs) -> NoReturn:
    last_traceback = traceback.format_exc()
    failure = {}
    if params is not None:
        failure['params'] = params
    if isinstance(exception, APIException):
        failure['message'] = exception.message
        failure['code'] = exception.code
        failure['details'] = exception.details
    elif isinstance(exception, ActionException):
        failure['action'] = {k: getattr(exception.action, k) for k in exception.action.__slots__}
    exception_message = to_native(exception)
    if msg is not None:
        msg = f'{exception_message}: {msg}'
    else:
        msg = exception_message
    self.module.fail_json(msg=msg, exception=last_traceback, failure=failure, **kwargs)