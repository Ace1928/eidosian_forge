from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware import connect_to_api
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_all_categories(self):
    """Retrieve all category information."""
    try:
        for category in self.category_service.list():
            category_obj = self.category_service.get(category)
            self.global_categories[category_obj.name] = dict(category_description=category_obj.description, category_used_by=category_obj.used_by, category_cardinality=str(category_obj.cardinality), category_associable_types=category_obj.associable_types, category_id=category_obj.id, category_name=category_obj.name)
    except Error as error:
        self.module.fail_json(msg=self.get_error_message(error))
    except Exception as exc_err:
        self.module.fail_json(msg=to_native(exc_err))