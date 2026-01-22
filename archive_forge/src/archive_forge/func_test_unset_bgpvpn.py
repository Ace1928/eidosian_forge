import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_unset_bgpvpn(self):
    attrs = {'route_targets': ['unset_rt1', 'unset_rt2', 'unset_rt3'], 'import_targets': ['unset_irt1', 'unset_irt2', 'unset_irt3'], 'export_targets': ['unset_ert1', 'unset_ert2', 'unset_ert3'], 'route_distinguishers': ['unset_rd1', 'unset_rd2', 'unset_rd3']}
    fake_bgpvpn = fakes.create_one_bgpvpn(attrs)
    self.networkclient.get_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
    self.networkclient.update_bgpvpn = mock.Mock()
    arglist = [fake_bgpvpn['id'], '--route-target', 'unset_rt1', '--import-target', 'unset_irt1', '--export-target', 'unset_ert1', '--route-distinguisher', 'unset_rd1']
    verifylist = [('bgpvpn', fake_bgpvpn['id']), ('route_targets', ['unset_rt1']), ('purge_route_target', False), ('import_targets', ['unset_irt1']), ('purge_import_target', False), ('export_targets', ['unset_ert1']), ('purge_export_target', False), ('route_distinguishers', ['unset_rd1']), ('purge_route_distinguisher', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'route_targets': list(set(fake_bgpvpn['route_targets']) - set(['unset_rt1'])), 'import_targets': list(set(fake_bgpvpn['import_targets']) - set(['unset_irt1'])), 'export_targets': list(set(fake_bgpvpn['export_targets']) - set(['unset_ert1'])), 'route_distinguishers': list(set(fake_bgpvpn['route_distinguishers']) - set(['unset_rd1']))}
    self.networkclient.update_bgpvpn.assert_called_once_with(fake_bgpvpn['id'], **attrs)
    self.assertIsNone(result)