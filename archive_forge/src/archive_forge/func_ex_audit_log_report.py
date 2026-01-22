import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_audit_log_report(self, start_date, end_date):
    """
        Get audit log report

        :param start_date: Start date for the report
        :type  start_date: ``str`` in format YYYY-MM-DD

        :param end_date: End date for the report
        :type  end_date: ``str`` in format YYYY-MM-DD

        :rtype: ``list`` of ``list``
        """
    result = self.connection.raw_request_with_orgId_api_1('auditlog?startDate={}&endDate={}'.format(start_date, end_date))
    return self._format_csv(result.response)