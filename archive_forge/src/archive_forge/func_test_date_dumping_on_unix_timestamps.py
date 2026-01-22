from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_date_dumping_on_unix_timestamps(self):
    result = _dump_date(1331755274.59, ' ')
    assert result == 'Wed, 14 Mar 2012 20:01:14 GMT'