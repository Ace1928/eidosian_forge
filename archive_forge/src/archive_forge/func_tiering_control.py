from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def tiering_control(self, current):
    """return whether the backend meets FabricPool requirements:
            required: all aggregates are in a FabricPool
            disallowed: all aggregates are not in a FabricPool
            best_effort: mixed
        """
    fabricpools = [self.is_fabricpool(aggregate['name'], aggregate['uuid']) for aggregate in current.get('aggregates', [])]
    if not fabricpools:
        return None
    if all(fabricpools):
        return 'required'
    if any(fabricpools):
        return 'best_effort'
    return 'disallowed'