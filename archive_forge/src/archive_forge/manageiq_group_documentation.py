from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
 Creates the ansible result object from a manageiq group entity

        Returns:
            a dict with the group id, description, role, tenant, filters, group_type, created_on, updated_on
        