import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list_with_sort_key_dir(self):
    expect = [('GET', self.base_path + '?sort_key=uuid&sort_dir=desc', {}, None)]
    self._test_nodegroup_list_with_filters(self.cluster_id, sort_key='uuid', sort_dir='desc', expect=expect)