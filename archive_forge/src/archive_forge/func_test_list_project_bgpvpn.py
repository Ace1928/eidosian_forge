import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_list_project_bgpvpn(self):
    count = 3
    project_id = 'list_fake_project_id'
    attrs = {'tenant_id': project_id}
    fake_bgpvpns = fakes.create_bgpvpns(count=count, attrs=attrs)
    self.networkclient.bgpvpns = mock.Mock(return_value=fake_bgpvpns)
    arglist = ['--project', project_id]
    verifylist = [('project', project_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.networkclient.bgpvpns.assert_called_once_with(tenant_id=project_id)
    self.assertEqual(headers, list(headers_short))
    self.assertListItemEqual(list(data), [_get_data(fake_bgpvpn, columns_short) for fake_bgpvpn in fake_bgpvpns])