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
def ex_update_ptr_record(self, record, domain=None, extra=None):
    """
        Update a PTR record for a specific IP on a specific device.

        If you need to change the domain or ttl, use this API to
        update the record by deleting the old one and creating a new one.

        :param record: the original :class:`RackspacePTRRecord`
        :param domain: the fqdn you want that IP to represent
        :param extra: a ``dict`` with optional extra values:
            ttl - the time-to-live of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
    if domain is not None and domain == record.domain:
        domain = None
    if extra is not None:
        extra = dict(extra)
        for key in extra:
            if key in record.extra and record.extra[key] == extra[key]:
                del extra[key]
    if domain is None and (not extra):
        return record
    _check_ptr_extra_fields(record)
    ip = record.ip
    self.ex_delete_ptr_record(record)
    return self.ex_create_ptr_record(record, ip, domain, extra=extra)