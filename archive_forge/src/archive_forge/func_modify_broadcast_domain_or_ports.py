from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_broadcast_domain_or_ports(self, modify, current=None):
    """
        :param modify: modify attributes.
        """
    modify_keys = list(modify.keys())
    domain_modify_options = ['mtu', 'name', 'ipspace']
    if any((x in modify_keys for x in domain_modify_options)):
        if self.use_rest:
            if modify.get('ports'):
                del modify['ports']
            self.modify_broadcast_domain_rest(current['uuid'], modify)
            if modify.get('ipspace'):
                current['ipspace'] = modify['ipspace']
        else:
            self.modify_broadcast_domain()
    if 'ports' in modify_keys:
        self.modify_broadcast_domain_ports(current)