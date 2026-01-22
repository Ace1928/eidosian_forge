from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import metrics
from monascaclient.v2_0 import shell
def test_bad_metrics(self):
    raw_args_list = [['metric1'], ['123'], ['']]
    name, cmd_clazz = migr.create_command_class('do_metric_create', shell)
    for raw_args in raw_args_list:
        cmd = cmd_clazz(mock.Mock(), mock.Mock())
        parser = cmd.get_parser(name)
        self.assertRaises(SystemExit, parser.parse_args, raw_args)