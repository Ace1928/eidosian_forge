from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_api_details(self):
    """ Get api details of the array """
    try:
        LOG.info('Getting API details ')
        api_version = self.powerflex_conn.system.api_version()
        return api_version
    except Exception as e:
        msg = 'Get API details from Powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)