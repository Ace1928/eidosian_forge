import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_watch(watch):
    updated_time = heat_timeutils.isotime(watch.updated_at or timeutils.utcnow())
    result = {rpc_api.WATCH_ACTIONS_ENABLED: watch.rule.get(rpc_api.RULE_ACTIONS_ENABLED), rpc_api.WATCH_ALARM_ACTIONS: watch.rule.get(rpc_api.RULE_ALARM_ACTIONS), rpc_api.WATCH_TOPIC: watch.rule.get(rpc_api.RULE_TOPIC), rpc_api.WATCH_UPDATED_TIME: updated_time, rpc_api.WATCH_DESCRIPTION: watch.rule.get(rpc_api.RULE_DESCRIPTION), rpc_api.WATCH_NAME: watch.name, rpc_api.WATCH_COMPARISON: watch.rule.get(rpc_api.RULE_COMPARISON), rpc_api.WATCH_DIMENSIONS: watch.rule.get(rpc_api.RULE_DIMENSIONS) or [], rpc_api.WATCH_PERIODS: watch.rule.get(rpc_api.RULE_PERIODS), rpc_api.WATCH_INSUFFICIENT_ACTIONS: watch.rule.get(rpc_api.RULE_INSUFFICIENT_ACTIONS), rpc_api.WATCH_METRIC_NAME: watch.rule.get(rpc_api.RULE_METRIC_NAME), rpc_api.WATCH_NAMESPACE: watch.rule.get(rpc_api.RULE_NAMESPACE), rpc_api.WATCH_OK_ACTIONS: watch.rule.get(rpc_api.RULE_OK_ACTIONS), rpc_api.WATCH_PERIOD: watch.rule.get(rpc_api.RULE_PERIOD), rpc_api.WATCH_STATE_REASON: watch.rule.get(rpc_api.RULE_STATE_REASON), rpc_api.WATCH_STATE_REASON_DATA: watch.rule.get(rpc_api.RULE_STATE_REASON_DATA), rpc_api.WATCH_STATE_UPDATED_TIME: heat_timeutils.isotime(watch.rule.get(rpc_api.RULE_STATE_UPDATED_TIME, timeutils.utcnow())), rpc_api.WATCH_STATE_VALUE: watch.state, rpc_api.WATCH_STATISTIC: watch.rule.get(rpc_api.RULE_STATISTIC), rpc_api.WATCH_THRESHOLD: watch.rule.get(rpc_api.RULE_THRESHOLD), rpc_api.WATCH_UNIT: watch.rule.get(rpc_api.RULE_UNIT), rpc_api.WATCH_STACK_ID: watch.stack_id}
    return result