import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def export_zone_to_bind_format(self, zone):
    action = '{}/domains/{}/records'.format(API_BASE, zone.id)
    headers = {'Accept': 'text/plain'}
    resp = self.connection.request(action=action, method='GET', headers=headers, raw=True)
    return resp.body