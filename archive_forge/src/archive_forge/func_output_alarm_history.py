import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def output_alarm_history(args, alarm_history):
    if args.json:
        print(utils.json_formatter(alarm_history))
        return
    cols = ['alarm_id', 'new_state', 'old_state', 'reason', 'reason_data', 'metric_name', 'metric_dimensions', 'timestamp']
    formatters = {'alarm_id': lambda x: x['alarm_id'], 'new_state': lambda x: x['new_state'], 'old_state': lambda x: x['old_state'], 'reason': lambda x: x['reason'], 'reason_data': lambda x: x['reason_data'], 'metric_name': lambda x: format_metric_name(x['metrics']), 'metric_dimensions': lambda x: format_metric_dimensions(x['metrics']), 'timestamp': lambda x: x['timestamp']}
    if isinstance(alarm_history, list):
        utils.print_list(alarm_history, cols, formatters=formatters)
    else:
        alarm_list = list()
        alarm_list.append(alarm_history)
        utils.print_list(alarm_list, cols, formatters=formatters)