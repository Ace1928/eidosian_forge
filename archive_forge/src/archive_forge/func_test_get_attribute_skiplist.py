import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_get_attribute_skiplist(self):
    resg = self._create_dummy_stack()
    resg.data = mock.Mock(return_value={'name_blacklist': '3,5'})
    expected = ['3', '5']
    self.assertEqual(expected, resg.FnGetAtt(resg.REMOVED_RSRC_LIST))