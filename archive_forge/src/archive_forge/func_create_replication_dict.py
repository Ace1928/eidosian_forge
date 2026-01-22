from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_replication_dict(replication):
    if replication is None:
        return None
    results = dict(id=replication.id, name=replication.name, location=replication.location, provisioning_state=replication.provisioning_state, tags=replication.tags, status=replication.status.display_status)
    return results