import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def get_share_export_locations(self, share):
    cmd = f'export location list {share}'
    export_locations = json.loads(self.openstack(f'share {cmd} -f json'))
    return export_locations