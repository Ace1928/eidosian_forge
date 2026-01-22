import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def share_group_type_access_delete(self, group_type, access_id):
    cmd = f'group type access delete {group_type} {access_id} '
    self.dict_result('share', cmd)