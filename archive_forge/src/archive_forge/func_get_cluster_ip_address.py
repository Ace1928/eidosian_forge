from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_ip_address(self, cluster_ip_address, ignore_error=True):
    """ get node information if it is discoverable
            return:
                None if the cluster cannot be reached
                a dictionary of attributes
        """
    if cluster_ip_address is None:
        return None
    if self.use_rest:
        nodes = self.get_cluster_ip_addresses_rest(cluster_ip_address)
    else:
        nodes = self.get_cluster_ip_addresses(cluster_ip_address, ignore_error=ignore_error)
    return nodes if len(nodes) > 0 else None