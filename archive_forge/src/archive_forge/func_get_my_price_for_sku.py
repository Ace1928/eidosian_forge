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
@requires(['MarketplaceId', 'SellerSKUList'])
@structured_lists('SellerSKUList.SellerSKU')
@api_action('Products', 20, 10, 'GetMyPriceForSKU')
def get_my_price_for_sku(self, request, response, **kw):
    """Returns pricing information for your own offer listings, based on SellerSKU.
        """
    return self._post_request(request, kw, response)