import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_list_with_sort_key_dir(self):
    expect = [('GET', '/v1/clustertemplates/?sort_key=uuid&sort_dir=desc', {}, None)]
    self._test_clustertemplate_list_with_filters(sort_key='uuid', sort_dir='desc', expect=expect)