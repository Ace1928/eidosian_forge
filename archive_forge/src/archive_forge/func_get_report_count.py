from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
@boolean_arguments('Acknowledged')
@structured_lists('ReportTypeList.Type')
@api_action('Reports', 10, 45)
def get_report_count(self, request, response, **kw):
    """Returns a count of the reports, created in the previous 90 days,
           with a status of _DONE_ and that are available for download.
        """
    return self._post_request(request, kw, response)