from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def motd_get_iter(self):
    """
        Compose NaElement object to query current motd
        :return: NaElement object for vserver-motd-get-iter
        """
    motd_get_iter = netapp_utils.zapi.NaElement('vserver-motd-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    motd_info = netapp_utils.zapi.NaElement('vserver-motd-info')
    motd_info.add_new_child('is-cluster-message-enabled', str(self.parameters['show_cluster_motd']))
    motd_info.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(motd_info)
    motd_get_iter.add_child_elem(query)
    return motd_get_iter