import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_watch_data(wd, rule_names):
    namespace = wd.data['Namespace']
    metric = [(k, v) for k, v in wd.data.items() if k != 'Namespace']
    if len(metric) == 1:
        metric_name, metric_data = metric[0]
    else:
        LOG.error('Unexpected number of keys in watch_data.data!')
        return
    result = {rpc_api.WATCH_DATA_ALARM: rule_names.get(wd.watch_rule_id), rpc_api.WATCH_DATA_METRIC: metric_name, rpc_api.WATCH_DATA_TIME: heat_timeutils.isotime(wd.created_at), rpc_api.WATCH_DATA_NAMESPACE: namespace, rpc_api.WATCH_DATA: metric_data}
    return result