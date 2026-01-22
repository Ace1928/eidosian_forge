from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_service_templates_list(self):
    """ Get the list of service templates on a given PowerFlex Manager system """
    try:
        LOG.info('Getting service templates list ')
        service_templates = self.powerflex_conn.service_template.get(filters=self.populate_filter_list(), sort=self.get_param_value('sort'), offset=self.get_param_value('offset'), limit=self.get_param_value('limit'), full=self.get_param_value('full'), include_attachments=self.get_param_value('include_attachments'))
        return service_templates
    except Exception as e:
        msg = f'Get service templates from PowerFlex Manager failed with error {str(e)}'
        return self.handle_error_exit(msg)