import argparse
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from testtools import matchers
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_cacert(self):
    cacert = '/path/to/cacert'
    s = self.get_session('--os-cacert %s' % cacert)
    self.assertEqual(cacert, s.verify)