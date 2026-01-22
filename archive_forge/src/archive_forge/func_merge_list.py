import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def merge_list(old, new):
    """merges lists and comma delimited lists."""
    if not old:
        return new
    if isinstance(new, list):
        old.extend(new)
        return old
    else:
        return ','.join([old, new])