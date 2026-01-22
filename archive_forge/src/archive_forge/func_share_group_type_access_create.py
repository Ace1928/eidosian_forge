import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def share_group_type_access_create(self, group_type, project):
    cmd = f'group type access create {group_type} {project} '
    self.dict_result('share', cmd)