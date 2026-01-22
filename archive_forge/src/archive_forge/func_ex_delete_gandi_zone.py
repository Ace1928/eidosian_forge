import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def ex_delete_gandi_zone(self, zone_uuid):
    self.connection.request(action='{}/zones/{}'.format(API_BASE, zone_uuid), method='DELETE')
    return True