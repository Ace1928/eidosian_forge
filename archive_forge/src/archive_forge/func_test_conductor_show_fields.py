import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_show_fields(self):
    arglist = ['xxxxx', '--fields', 'hostname', 'alive']
    verifylist = [('conductor', 'xxxxx'), ('fields', [['hostname', 'alive']])]
    fake_cond = copy.deepcopy(baremetal_fakes.CONDUCTOR)
    fake_cond.pop('conductor_group')
    fake_cond.pop('drivers')
    self.baremetal_mock.conductor.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_cond, loaded=True)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertNotIn('conductor_group', columns)
    args = ['xxxxx']
    fields = ['hostname', 'alive']
    self.baremetal_mock.conductor.get.assert_called_with(*args, fields=fields)