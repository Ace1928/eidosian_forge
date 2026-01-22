from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import alarm_definitions as ad
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_should_update(self, mc):
    mc.return_value = c = FakeV2Client()
    ad_id = '0495340b-58fd-4e1c-932b-5e6f9cc96490'
    ad_name = 'alarm_name'
    ad_desc = 'test_alarm_definition'
    ad_expr = 'avg(Test_Metric_1)>=10'
    ad_action_id = '16012650-0b62-4692-9103-2d04fe81cc93'
    ad_action_enabled = 'True'
    ad_match_by = 'hostname'
    ad_severity = 'CRITICAL'
    raw_args = [ad_id, ad_name, ad_desc, ad_expr, ad_action_id, ad_action_id, ad_action_id, ad_action_enabled, ad_match_by, ad_severity]
    name, cmd_clazz = migr.create_command_class('do_alarm_definition_update', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    c.alarm_definitions.update.assert_called_once_with(actions_enabled=True, alarm_actions=[ad_action_id], alarm_id=ad_id, description=ad_desc, expression=ad_expr, match_by=[ad_match_by], name=ad_name, ok_actions=[ad_action_id], severity=ad_severity, undetermined_actions=[ad_action_id])