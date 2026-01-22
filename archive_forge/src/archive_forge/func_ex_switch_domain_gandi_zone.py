import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def ex_switch_domain_gandi_zone(self, domain, zone_uuid):
    domain_data = {'zone_uuid': zone_uuid}
    self.connection.request(action='{}/domains/{}'.format(API_BASE, domain), method='PATCH', data=domain_data)
    return True