from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def compare_tenant(self, tenant, name, description):
    """ Compare tenant fields with new field values.

        Returns:
            false if tenant fields have some difference from new fields, true o/w.
        """
    found_difference = name and tenant['name'] != name or (description and tenant['description'] != description)
    return not found_difference