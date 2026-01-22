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
@requires(['MarketplaceId', 'IdType', 'IdList'])
@structured_lists('IdList.Id')
@api_action('Products', 20, 20)
def get_matching_product_for_id(self, request, response, **kw):
    """Returns a list of products and their attributes, based on
           a list of Product IDs that you specify.
        """
    return self._post_request(request, kw, response)