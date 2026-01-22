import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
@utils.arg('--alarm-definition-id', metavar='<ALARM_DEFINITION_ID>', help='The ID of the alarm definition.')
@utils.arg('--metric-name', metavar='<METRIC_NAME>', help='Name of the metric.')
@utils.arg('--metric-dimensions', metavar='<KEY1=VALUE1,KEY2,KEY3=VALUE2...>', help='key value pair used to specify a metric dimension or just key to select all values of that dimension.This can be specified multiple times, or once with parameters separated by a comma. Dimensions need quoting when they contain special chars [&,(,),{,},>,<] that confuse the CLI parser.', action='append')
@utils.arg('--state', metavar='<ALARM_STATE>', help='ALARM_STATE is one of [UNDETERMINED, OK, ALARM].')
@utils.arg('--severity', metavar='<SEVERITY>', help='Severity is one of ["LOW", "MEDIUM", "HIGH", "CRITICAL"].')
@utils.arg('--state-updated-start-time', metavar='<UTC_STATE_UPDATED_START>', help='Return all alarms whose state was updated on or after the time specified.')
@utils.arg('--lifecycle-state', metavar='<LIFECYCLE_STATE>', help='The lifecycle state of the alarm.')
@utils.arg('--link', metavar='<LINK>', help='The link to external data associated with the alarm.')
@utils.arg('--sort-by', metavar='<SORT BY FIELDS>', help='Fields to sort by as a comma separated list. Valid values are alarm_id, alarm_definition_id, state, severity, lifecycle_state, link, state_updated_timestamp, updated_timestamp, created_timestamp. Fields may be followed by "asc" or "desc", ex "severity desc", to set the direction of sorting.')
@utils.arg('--offset', metavar='<OFFSET LOCATION>', help='The offset used to paginate the return data.')
@utils.arg('--limit', metavar='<RETURN LIMIT>', help='The amount of data to be returned up to the API maximum limit.')
def do_alarm_list(mc, args):
    """List alarms for this tenant."""
    fields = {}
    if args.alarm_definition_id:
        fields['alarm_definition_id'] = args.alarm_definition_id
    if args.metric_name:
        fields['metric_name'] = args.metric_name
    if args.metric_dimensions:
        fields['metric_dimensions'] = utils.format_dimensions_query(args.metric_dimensions)
    if args.state:
        if args.state.upper() not in state_types:
            errmsg = 'Invalid state, not one of [' + ', '.join(state_types) + ']'
            print(errmsg)
            return
        fields['state'] = args.state
    if args.severity:
        if not _validate_severity(args.severity):
            return
        fields['severity'] = args.severity
    if args.state_updated_start_time:
        fields['state_updated_start_time'] = args.state_updated_start_time
    if args.lifecycle_state:
        fields['lifecycle_state'] = args.lifecycle_state
    if args.link:
        fields['link'] = args.link
    if args.limit:
        fields['limit'] = args.limit
    if args.offset:
        fields['offset'] = args.offset
    if args.sort_by:
        sort_by = args.sort_by.split(',')
        for field in sort_by:
            field_values = field.lower().split()
            if len(field_values) > 2:
                print('Invalid sort_by value {}'.format(field))
            if field_values[0] not in allowed_alarm_sort_by:
                print('Sort-by field name {} is not in [{}]'.format(field_values[0], allowed_alarm_sort_by))
                return
            if len(field_values) > 1 and field_values[1] not in ['asc', 'desc']:
                print('Invalid value {}, must be asc or desc'.format(field_values[1]))
        fields['sort_by'] = args.sort_by
    try:
        alarm = mc.alarms.list(**fields)
    except (osc_exc.ClientException, k_exc.HttpError) as he:
        raise osc_exc.CommandError('%s\n%s' % (he.message, he.details))
    else:
        if args.json:
            print(utils.json_formatter(alarm))
            return
        cols = ['id', 'alarm_definition_id', 'alarm_definition_name', 'metric_name', 'metric_dimensions', 'severity', 'state', 'lifecycle_state', 'link', 'state_updated_timestamp', 'updated_timestamp', 'created_timestamp']
        formatters = {'id': lambda x: x['id'], 'alarm_definition_id': lambda x: x['alarm_definition']['id'], 'alarm_definition_name': lambda x: x['alarm_definition']['name'], 'metric_name': lambda x: format_metric_name(x['metrics']), 'metric_dimensions': lambda x: format_metric_dimensions(x['metrics']), 'severity': lambda x: x['alarm_definition']['severity'], 'state': lambda x: x['state'], 'lifecycle_state': lambda x: x['lifecycle_state'], 'link': lambda x: x['link'], 'state_updated_timestamp': lambda x: x['state_updated_timestamp'], 'updated_timestamp': lambda x: x['updated_timestamp'], 'created_timestamp': lambda x: x['created_timestamp']}
        if isinstance(alarm, list):
            utils.print_list(alarm, cols, formatters=formatters)
        else:
            alarm_list = list()
            alarm_list.append(alarm)
            utils.print_list(alarm_list, cols, formatters=formatters)