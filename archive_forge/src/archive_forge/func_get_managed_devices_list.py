from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_managed_devices_list(self):
    """ Get the list of managed devices on a given PowerFlex Manager system """
    try:
        LOG.info('Getting managed devices list ')
        devices = self.powerflex_conn.managed_device.get(filters=self.populate_filter_list(), limit=self.get_param_value('limit'), offset=self.get_param_value('offset'), sort=self.get_param_value('sort'))
        return devices
    except Exception as e:
        msg = f'Get managed devices from PowerFlex Manager failed with error {str(e)}'
        return self.handle_error_exit(msg)