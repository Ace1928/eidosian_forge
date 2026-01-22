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
def has_completed(self, response):
    status = response.object['status']
    if status == 'ERROR':
        data = response.object['error']
        if 'code' and 'message' in data:
            message = '{} - {} ({})'.format(data['code'], data['message'], data['details'])
        else:
            message = data['message']
        raise LibcloudError(message, driver=self.driver)
    return status == 'COMPLETED'