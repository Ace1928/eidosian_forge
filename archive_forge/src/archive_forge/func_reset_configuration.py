from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, get_all_objs, wait_for_task, PyVmomi
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
def reset_configuration(self):
    if not self.host.runtime.inMaintenanceMode:
        self.enter_maintenance()
    try:
        self.host.configManager.firmwareSystem.ResetFirmwareToFactoryDefaults()
        self.module.exit_json(changed=True)
    except Exception as e:
        self.exit_maintenance()
        self.module.fail_json(msg=to_native(e))