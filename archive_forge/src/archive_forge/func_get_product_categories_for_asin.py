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
@requires(['MarketplaceId', 'ASIN'])
@api_action('Products', 20, 20, 'GetProductCategoriesForASIN')
def get_product_categories_for_asin(self, request, response, **kw):
    """Returns the product categories that an ASIN belongs to.
        """
    return self._post_request(request, kw, response)