from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def handle_error_exit(self, detailed_message):
    match = re.search("displayMessage=([^']+)", detailed_message)
    error_message = match.group(1) if match else detailed_message
    LOG.error(error_message)
    if re.search(ERROR_CODES, detailed_message):
        return []
    self.module.fail_json(msg=error_message)