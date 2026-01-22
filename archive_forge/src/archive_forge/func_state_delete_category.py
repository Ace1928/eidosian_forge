from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware import connect_to_api
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_delete_category(self):
    """Delete category."""
    category_id = self.global_categories[self.category_name]['category_id']
    try:
        self.category_service.delete(category_id=category_id)
    except Error as error:
        self.module.fail_json(msg='%s' % self.get_error_message(error))
    self.module.exit_json(changed=True, category_results=dict(msg="Category '%s' deleted." % self.category_name, category_id=category_id))