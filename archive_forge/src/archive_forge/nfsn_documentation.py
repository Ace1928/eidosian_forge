import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
from libcloud.common.exceptions import BaseHTTPError

        Use this method to delete a record.

        :param record: record to delete
        :type record: `Record`

        :rtype: Bool
        