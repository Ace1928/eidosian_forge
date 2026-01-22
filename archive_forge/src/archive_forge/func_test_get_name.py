import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
def test_get_name(self):
    self.assertEqual('loginsight', self._driver.get_name())