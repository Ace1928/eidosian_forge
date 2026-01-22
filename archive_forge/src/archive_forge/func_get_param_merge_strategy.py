import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def get_param_merge_strategy(merge_strategies, param_key, available_strategies=None):
    if not available_strategies:
        available_strategies = {}
    if merge_strategies is None:
        return OVERWRITE
    env_default = merge_strategies.get('default', OVERWRITE)
    merge_strategy = merge_strategies.get(param_key, available_strategies.get(param_key, env_default))
    if merge_strategy in ALLOWED_PARAM_MERGE_STRATEGIES:
        return merge_strategy
    return env_default