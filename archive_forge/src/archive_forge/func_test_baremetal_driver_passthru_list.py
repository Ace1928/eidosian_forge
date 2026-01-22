import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_passthru_list(self):
    arglist = ['fakedrivername']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['fakedrivername']
    self.baremetal_mock.driver.get_vendor_passthru_methods.assert_called_with(*args)
    collist = ('Name', 'Supported HTTP methods', 'Async', 'Description', 'Response is attachment')
    self.assertEqual(collist, tuple(columns))
    datalist = (('lookup', 'POST', 'false', '', 'false'),)
    self.assertEqual(datalist, tuple(data))