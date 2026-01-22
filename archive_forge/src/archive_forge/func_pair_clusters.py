from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def pair_clusters(self):
    """
            Start cluster pairing on source, and complete on target cluster
        """
    try:
        pair_key = self.elem.start_cluster_pairing()
        self.dest_elem.complete_cluster_pairing(cluster_pairing_key=pair_key.cluster_pairing_key)
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error pairing cluster %s and %s' % (self.parameters['hostname'], self.parameters['dest_mvip']), exception=to_native(err))