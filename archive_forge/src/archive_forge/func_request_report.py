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
@requires(['ReportType'])
@structured_lists('MarketplaceIdList.Id')
@boolean_arguments('ReportOptions=ShowSalesChannel')
@api_action('Reports', 15, 60)
def request_report(self, request, response, **kw):
    """Creates a report request and submits the request to Amazon MWS.
        """
    return self._post_request(request, kw, response)