import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_list_with_detail(self):
    expect = [('GET', '/v1/clustertemplates/detail', {}, None)]
    self._test_clustertemplate_list_with_filters(detail=True, expect=expect)