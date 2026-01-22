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
@requires(['AmazonOrderReferenceId'])
@api_action('OffAmazonPayments', 20, 2)
def get_order_reference_details(self, request, response, **kw):
    """Returns details about the Order Reference object and its current
           state.
        """
    return self._post_request(request, kw, response)