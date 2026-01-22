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
@api_action('Reports', 10, 45)
def cancel_report_requests(self, request, response, **kw):
    """Cancel one or more report requests, returning the count of the
           canceled report requests and the report request information.
        """
    return self._post_request(request, kw, response)