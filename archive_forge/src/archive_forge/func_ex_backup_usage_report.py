import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_backup_usage_report(self, start_date, end_date, location):
    """
        Get audit log report

        :param start_date: Start date for the report
        :type  start_date: ``str`` in format YYYY-MM-DD

        :param end_date: End date for the report
        :type  end_date: ``str`` in format YYYY-MM-DD

        :keyword location: Filters the node list to nodes that are
                           located in this location
        :type    location: :class:`NodeLocation` or ``str``

        :rtype: ``list`` of ``list``
        """
    datacenter_id = self._location_to_location_id(location)
    result = self.connection.raw_request_with_orgId_api_1('backup/detailedUsageReport?datacenterId=%s&fromDate=%s&toDate=%s' % (datacenter_id, start_date, end_date))
    return self._format_csv(result.response)