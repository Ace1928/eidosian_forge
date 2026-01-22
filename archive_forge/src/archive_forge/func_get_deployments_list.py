from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_deployments_list(self):
    """ Get the list of deployments on a given PowerFlex Manager system """
    try:
        LOG.info('Getting deployments list ')
        deployments = self.powerflex_conn.deployment.get(filters=self.populate_filter_list(), sort=self.get_param_value('sort'), limit=self.get_param_value('limit'), offset=self.get_param_value('offset'), include_devices=self.get_param_value('include_devices'), include_template=self.get_param_value('include_template'), full=self.get_param_value('full'))
        return deployments
    except Exception as e:
        msg = f'Get deployments from PowerFlex Manager failed with error {str(e)}'
        return self.handle_error_exit(msg)