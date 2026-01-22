import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def check_create_network_subnet(self, share_network, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, restart_check=None):
    cmd = f'network subnet create {share_network} --check-only'
    if neutron_net_id:
        cmd += f' --neutron-net-id {neutron_net_id}'
    if neutron_subnet_id:
        cmd += f' --neutron-subnet-id {neutron_subnet_id}'
    if availability_zone:
        cmd += f' --availability-zone {availability_zone}'
    if restart_check:
        cmd += ' --restart-check'
    check_result = self.dict_result('share', cmd)
    return check_result