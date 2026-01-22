import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
def ex_delete_ptr_record(self, record):
    """
        Delete an existing PTR Record

        :param record: the original :class:`RackspacePTRRecord`
        :rtype: ``bool``
        """
    _check_ptr_extra_fields(record)
    self.connection.set_context({'resource': 'record', 'id': record.id})
    self.connection.async_request(action='/rdns/%s' % record.extra['service_name'], method='DELETE', params={'href': record.extra['uri'], 'ip': record.ip})
    return True