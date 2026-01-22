from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_datacenter_by_name, find_cluster_by_name, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_all_cluster_objs(self, parent):
    """
        Get all cluster managed objects from given parent object
        Args:
            parent: Managed objected of datacenter or host folder

        Returns: List of host managed objects

        """
    cluster_objs = []
    if isinstance(parent, vim.Datacenter):
        folder = parent.hostFolder
    else:
        folder = parent
    for child in folder.childEntity:
        if isinstance(child, vim.Folder):
            cluster_objs = cluster_objs + self.get_all_cluster_objs(child)
        if isinstance(child, vim.ClusterComputeResource):
            cluster_objs.append(child)
    return cluster_objs