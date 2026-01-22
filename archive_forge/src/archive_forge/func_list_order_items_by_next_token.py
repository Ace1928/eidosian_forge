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
@requires(['NextToken'])
@api_action('Orders', 30, 2)
def list_order_items_by_next_token(self, request, response, **kw):
    """Returns the next page of order items using the NextToken
           value that was returned by your previous request to either
           ListOrderItems or ListOrderItemsByNextToken.
        """
    return self._post_request(request, kw, response)