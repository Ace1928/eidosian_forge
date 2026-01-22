from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_marketplace_participations(self):
    response = self.mws.list_marketplace_participations()
    result = response.ListMarketplaceParticipationsResult
    self.assertTrue(result.ListMarketplaces.Marketplace[0].MarketplaceId)