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
@requires(['SellerFulfillmentOrderId'])
@api_action('Outbound', 30, 0.5)
def get_fulfillment_order(self, request, response, **kw):
    """Returns a fulfillment order based on a specified
           SellerFulfillmentOrderId.
        """
    return self._post_request(request, kw, response)