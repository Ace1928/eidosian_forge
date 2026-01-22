from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible.module_utils._text import to_native
def get_datastore_cluster_children(self):
    """
        Return Datastore from the given datastore cluster object

        """
    return [ds for ds in self.datastore_cluster_obj.childEntity if isinstance(ds, vim.Datastore)]