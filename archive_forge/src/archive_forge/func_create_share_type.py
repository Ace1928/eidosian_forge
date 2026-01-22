import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def create_share_type(self, name=None, dhss=False, description=None, snapshot_support=None, create_share_from_snapshot_support=None, revert_to_snapshot_support=False, mount_snapshot_support=False, extra_specs={}, public=True, add_cleanup=True, client=None, formatter=None):
    name = name or data_utils.rand_name('autotest_share_type_name')
    cmd = f'create {name} {dhss} --public {public}'
    if description:
        cmd += f' --description {description}'
    if snapshot_support:
        cmd += f' --snapshot-support {snapshot_support}'
    if create_share_from_snapshot_support:
        cmd += f' --create-share-from-snapshot-support {create_share_from_snapshot_support}'
    if revert_to_snapshot_support:
        cmd += f' --revert-to-snapshot-support  {revert_to_snapshot_support}'
    if mount_snapshot_support:
        cmd += f' --mount-snapshot-support {mount_snapshot_support}'
    if extra_specs:
        specs = ' --extra-specs'
        for key, value in extra_specs.items():
            specs += f' {key}={value}'
        cmd += specs
    if formatter == 'json':
        cmd = f'share type {cmd} -f {formatter} '
        share_type = json.loads(self.openstack(cmd, client=client))
    else:
        share_type = self.dict_result('share type', cmd, client=client)
    if add_cleanup:
        self.addCleanup(self.openstack, f'share type delete {share_type['id']}')
    return share_type