import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
def test_bad_log(self):
    self.assertThat(lambda: LOG.warning('String %(p1)s %(p2)s', {'p1': 'something'}), matchers.raises(KeyError))