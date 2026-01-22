from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_rl_dict(module, array):
    rl_info = {}
    api_version = array._list_available_rest_versions()
    if ACTIVE_DR_API in api_version:
        try:
            rlinks = array.list_pod_replica_links()
            for rlink in range(0, len(rlinks)):
                link_name = rlinks[rlink]['local_pod_name']
                if rlinks[rlink]['recovery_point']:
                    since_epoch = rlinks[rlink]['recovery_point'] / 1000
                    recovery_datatime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_epoch))
                else:
                    recovery_datatime = None
                if rlinks[rlink]['lag']:
                    lag = str(rlinks[rlink]['lag'] / 1000) + 's'
                rl_info[link_name] = {'status': rlinks[rlink]['status'], 'direction': rlinks[rlink]['direction'], 'lag': lag, 'remote_pod_name': rlinks[rlink]['remote_pod_name'], 'remote_names': rlinks[rlink]['remote_names'], 'recovery_point': recovery_datatime}
        except Exception:
            module.warn('Replica Links info requires purestorage SDK 1.19 or hisher')
    return rl_info