from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_inbound_status(self):
    response = self.mws.get_inbound_service_status()
    status = response.GetServiceStatusResult.Status
    self.assertIn(status, ('GREEN', 'GREEN_I', 'YELLOW', 'RED'))