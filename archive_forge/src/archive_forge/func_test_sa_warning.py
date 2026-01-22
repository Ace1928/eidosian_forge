import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
def test_sa_warning(self):
    self.assertThat(lambda: warnings.warn('test sa warning error', exc.SAWarning), matchers.raises(exc.SAWarning))