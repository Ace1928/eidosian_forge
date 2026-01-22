from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def lookup_datastore_by_cluster(self):
    """ Get datastore(s) per cluster """
    cluster = find_cluster_by_name(self.content, self.params['cluster'])
    if not cluster:
        self.module.fail_json(msg='Failed to find cluster "%(cluster)s"' % self.params)
    c_dc = cluster.datastore
    return c_dc