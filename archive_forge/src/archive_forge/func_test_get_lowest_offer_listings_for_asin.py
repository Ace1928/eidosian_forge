from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_get_lowest_offer_listings_for_asin(self):
    asin = '144930544X'
    response = self.mws.get_lowest_offer_listings_for_asin(MarketplaceId=self.marketplace_id, ItemCondition='New', ASINList=[asin])
    listings = response._result[0].Product.LowestOfferListings
    self.assertTrue(len(listings.LowestOfferListing))