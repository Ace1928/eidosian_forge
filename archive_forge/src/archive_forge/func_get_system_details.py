from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def get_system_details(self):
    """Get system details
        :return: Details of PowerFlex system
        """
    try:
        resp = self.powerflex_conn.system.get()
        if len(resp) == 0:
            self.module.fail_json(msg='No system exist on the given host.')
        if len(resp) > 1:
            self.module.fail_json(msg='Multiple systems exist on the given host.')
        return resp[0]
    except Exception as e:
        msg = 'Failed to get system id with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)