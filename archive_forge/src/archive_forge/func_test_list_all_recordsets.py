from unittest import mock
from osc_lib.tests import utils
from designateclient.tests.osc import resources
from designateclient.v2 import base
from designateclient.v2.cli import recordsets
def test_list_all_recordsets(self):
    arg_list = ['all']
    verify_args = [('zone_id', 'all')]
    body = resources.load('recordset_list_all')
    result = base.DesignateList()
    result.extend(body['recordsets'])
    self.dns_client.recordsets.list_all_zones.return_value = result
    parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
    columns, data = self.cmd.take_action(parsed_args)
    results = list(data)
    self.assertEqual(5, len(results))