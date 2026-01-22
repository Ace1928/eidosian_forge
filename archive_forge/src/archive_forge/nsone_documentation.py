from libcloud.dns.base import Zone, Record, DNSDriver, RecordType
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneResponse, NsOneException, NsOneConnection

        Function to check if a `Zone` object exists.
        :param zone_id: ID of the `Zone` object.
        :type zone_id: ``str``

        :param zones_list: A list containing `Zone` objects.
        :type zones_list: ``list``.

        :rtype: Returns `True` or `False`.
        