import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_list(self):
    clustertemplates = self.mgr.list()
    expect = [('GET', '/v1/clustertemplates', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(clustertemplates, matchers.HasLength(2))