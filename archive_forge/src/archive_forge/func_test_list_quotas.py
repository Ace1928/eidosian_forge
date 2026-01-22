import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_list_quotas(self):
    quotas = self.mgr.list()
    expect = [('GET', '/v1/quotas', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(quotas, matchers.HasLength(1))