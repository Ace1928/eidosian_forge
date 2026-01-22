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
@requires(['ReportId'])
@api_action('Reports', 15, 60)
def get_report(self, request, response, **kw):
    """Returns the contents of a report.
        """
    return self._post_request(request, kw, response)