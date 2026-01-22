import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def create_share_transfer(self, share, name=None, client=None):
    name = name or data_utils.rand_name('autotest_share_transfer_name')
    cmd = f'transfer create {share} --name {name} '
    transfer_object = self.dict_result('share', cmd, client=client)
    return transfer_object