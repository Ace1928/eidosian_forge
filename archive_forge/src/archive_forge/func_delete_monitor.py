from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def delete_monitor(module):
    monitor = _get_monitor(module)
    if not monitor:
        module.exit_json(changed=False)
    try:
        msg = api.Monitor.delete(monitor['id'])
        module.exit_json(changed=True, msg=msg)
    except Exception as e:
        module.fail_json(msg=to_native(e), exception=traceback.format_exc())