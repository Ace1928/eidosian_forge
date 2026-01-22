from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import metrics
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_metric_create(self, mc):
    mc.return_value = c = FakeV2Client()
    raw_args = 'metric1 123 --time 1395691090'.split(' ')
    name, cmd_clazz = migr.create_command_class('do_metric_create', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    data = {'timestamp': 1395691090, 'name': 'metric1', 'value': 123.0}
    c.metrics.create.assert_called_once_with(**data)