from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_bucket_repl_dict(module, blade):
    bucket_repl_info = {}
    bucket_links = blade.bucket_replica_links.list_bucket_replica_links()
    for linkcnt in range(0, len(bucket_links.items)):
        bucket_name = bucket_links.items[linkcnt].local_bucket.name
        bucket_repl_info[bucket_name] = {'direction': bucket_links.items[linkcnt].direction, 'lag': bucket_links.items[linkcnt].lag, 'paused': bucket_links.items[linkcnt].paused, 'status': bucket_links.items[linkcnt].status, 'remote_bucket': bucket_links.items[linkcnt].remote_bucket.name, 'remote_credentials': bucket_links.items[linkcnt].remote_credentials.name, 'recovery_point': bucket_links.items[linkcnt].recovery_point, 'object_backlog': {}}
    api_version = blade.api_version.list_versions().versions
    if SMB_MODE_API_VERSION in api_version:
        blade = get_system(module)
        bucket_links = list(blade.get_bucket_replica_links().items)
        for linkcnt in range(0, len(bucket_links)):
            bucket_name = bucket_links[linkcnt].local_bucket.name
            bucket_repl_info[bucket_name]['object_backlog'] = {'bytes_count': bucket_links[linkcnt].object_backlog.bytes_count, 'delete_ops_count': bucket_links[linkcnt].object_backlog.delete_ops_count, 'other_ops_count': bucket_links[linkcnt].object_backlog.other_ops_count, 'put_ops_count': bucket_links[linkcnt].object_backlog.put_ops_count}
            bucket_repl_info[bucket_name]['cascading_enabled'] = bucket_links[linkcnt].cascading_enabled
    return bucket_repl_info