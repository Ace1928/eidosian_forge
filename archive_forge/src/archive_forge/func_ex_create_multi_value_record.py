import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def ex_create_multi_value_record(self, name, zone, type, data, extra=None):
    self._validate_record(None, name, type, data, extra)
    action = '{}/domains/{}/records'.format(API_BASE, zone.id)
    record_data = {'rrset_name': name, 'rrset_type': self.RECORD_TYPE_MAP[type], 'rrset_values': data}
    if extra is not None and 'ttl' in extra:
        record_data['rrset_ttl'] = extra['ttl']
    try:
        self.connection.request(action=action, method='POST', data=record_data)
    except ResourceConflictError:
        raise RecordAlreadyExistsError(value='', driver=self.connection.driver, record_id='{}:{}'.format(self.RECORD_TYPE_MAP[type], name))
    return self._to_record(record_data, zone)