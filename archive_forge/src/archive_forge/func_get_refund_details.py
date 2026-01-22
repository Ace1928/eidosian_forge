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
@requires(['AmazonRefundId'])
@api_action('OffAmazonPayments', 20, 2)
def get_refund_details(self, request, response, **kw):
    """Returns the status of a particular refund.
        """
    return self._post_request(request, kw, response)