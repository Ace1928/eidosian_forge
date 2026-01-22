import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
@staticmethod
def schema_from_outputs(json_snippet):
    if json_snippet:
        return dict(((k, Schema(v.get('Description'))) for k, v in json_snippet.items()))
    return {}