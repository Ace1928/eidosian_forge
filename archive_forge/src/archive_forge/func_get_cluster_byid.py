from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_cluster_byid(self, cluster_id):
    return self.conn.clusters.get(id=cluster_id)