import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('id', metavar='<ALARM_DEFINITION_ID>', help='The ID of the alarm definition.')
@utils.arg('--name', metavar='<ALARM_DEFINITION_NAME>', help='Name of the alarm definition.')
@utils.arg('--description', metavar='<DESCRIPTION>', help='Description of the alarm.')
@utils.arg('--expression', metavar='<EXPRESSION>', help='The alarm expression to evaluate. Quoted.')
@utils.arg('--alarm-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is ALARM. This param may be specified multiple times.', action='append')
@utils.arg('--ok-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is OK. This param may be specified multiple times.', action='append')
@utils.arg('--undetermined-actions', metavar='<NOTIFICATION-ID>', help='The notification method to use when an alarm state is UNDETERMINED. This param may be specified multiple times.', action='append')
@utils.arg('--actions-enabled', metavar='<ACTIONS-ENABLED>', help='The actions-enabled boolean is one of [true,false].')
@utils.arg('--severity', metavar='<SEVERITY>', help='Severity is one of [LOW, MEDIUM, HIGH, CRITICAL].')
def do_alarm_definition_patch(mc, args):
    """Patch the alarm definition."""
    fields = {}
    fields['alarm_id'] = args.id
    if args.name:
        fields['name'] = args.name
    if args.description:
        fields['description'] = args.description
    if args.expression:
        fields['expression'] = args.expression
    if args.alarm_actions:
        fields['alarm_actions'] = _arg_split_patch_update(args.alarm_actions, patch=True)
    if args.ok_actions:
        fields['ok_actions'] = _arg_split_patch_update(args.ok_actions, patch=True)
    if args.undetermined_actions:
        fields['undetermined_actions'] = _arg_split_patch_update(args.undetermined_actions, patch=True)
    if args.actions_enabled:
        if args.actions_enabled not in enabled_types:
            errmsg = 'Invalid value, not one of [' + ', '.join(enabled_types) + ']'
            print(errmsg)
            return
        fields['actions_enabled'] = args.actions_enabled in ['true', 'True']
    if args.severity:
        if not _validate_severity(args.severity):
            return
        fields['severity'] = args.severity
    try:
        alarm = mc.alarm_definitions.patch(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        print(jsonutils.dumps(alarm, indent=2))