import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def create_share_replica(self, share, availability_zone=None, share_network=None, wait=None, add_cleanup=True):
    cmd = f'replica create {share}'
    if availability_zone:
        cmd = cmd + f' --availability-zone {availability_zone}'
    if wait:
        cmd = cmd + ' --wait'
    if share_network:
        cmd = cmd + ' --share-network %s' % share_network
    replica_object = self.dict_result('share', cmd)
    self._wait_for_object_status('share replica', replica_object['id'], 'available')
    if add_cleanup:
        self.addCleanup(self.openstack, f'share replica delete {replica_object['id']} --wait')
    return replica_object