from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def unpair_clusters(self, pair_id_source, pair_id_dest):
    """
            Delete cluster pair
        """
    try:
        self.elem.remove_cluster_pair(cluster_pair_id=pair_id_source)
        self.dest_elem.remove_cluster_pair(cluster_pair_id=pair_id_dest)
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error unpairing cluster %s and %s' % (self.parameters['hostname'], self.parameters['dest_mvip']), exception=to_native(err))