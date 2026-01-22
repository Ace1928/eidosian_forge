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
@api_action('OffAmazonPayments', 10, 1)
def confirm_order_reference(self, request, response, **kw):
    """Confirms that the order reference is free of constraints and all
           required information has been set on the order reference.
        """
    return self._post_request(request, kw, response)