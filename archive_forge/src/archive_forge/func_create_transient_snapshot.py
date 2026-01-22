from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
import random
def create_transient_snapshot(self):
    snapshot_cmd = 'addsnapshot'
    snapshot_opts = {}
    random_number = ''.join(random.choices('0123456789', k=10))
    snapshot_name = f'snapshot_{random_number}'
    snapshot_opts['name'] = snapshot_name
    snapshot_opts['pool'] = self.module.params.get('pool', '')
    snapshot_opts['volumes'] = self.module.params.get('fromsourcevolumes', '')
    snapshot_opts['retentionminutes'] = 5
    self.restapi.svc_run_command(snapshot_cmd, snapshot_opts, cmdargs=None, timeout=10)
    return snapshot_name