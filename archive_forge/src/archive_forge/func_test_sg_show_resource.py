import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_sg_show_resource(self):
    self._create_sg('test')
    self.sg.client = mock.MagicMock()
    s_groups = mock.MagicMock()
    sg = mock.MagicMock()
    sg.to_dict.return_value = {'server_gr': 'info'}
    s_groups.get.return_value = sg
    self.sg.client().server_groups = s_groups
    self.assertEqual({'server_gr': 'info'}, self.sg.FnGetAtt('show'))
    s_groups.get.assert_called_once_with('test')