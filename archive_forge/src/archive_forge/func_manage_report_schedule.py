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
@requires(['ReportType', 'Schedule'])
@api_action('Reports', 10, 45)
def manage_report_schedule(self, request, response, **kw):
    """Creates, updates, or deletes a report request schedule for
           a specified report type.
        """
    return self._post_request(request, kw, response)