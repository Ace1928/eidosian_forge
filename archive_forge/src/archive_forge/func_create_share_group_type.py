import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def create_share_group_type(self, name=None, share_types=None, group_specs=None, public=True, add_cleanup=True):
    name = name or data_utils.rand_name('autotest_share_group_types_name')
    share_types = share_types or 'None'
    cmd = f'group type create {name} {share_types} '
    if group_specs:
        cmd = cmd + f' --group-specs {group_specs} '
    if not public:
        cmd = cmd + f' --public {public} '
    share_object = self.dict_result('share', cmd)
    if add_cleanup:
        self.addCleanup(self.openstack, 'share group type delete %s' % share_object['id'])
    return share_object