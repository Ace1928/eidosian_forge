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
@requires(['CustomerIdList'])
@structured_lists('CustomerIdList.CustomerId')
@api_action('CustomerInfo', 15, 12)
def get_customers_for_customer_id(self, request, response, **kw):
    """Returns a list of customer accounts based on search criteria that
           you specify.
        """
    return self._post_request(request, kw, response)