import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
@classmethod
def from_attribute(cls, schema_dict):
    """Return a Property Schema corresponding to a Attribute Schema."""
    msg = 'Old attribute schema is not supported'
    assert isinstance(schema_dict, cls), msg
    return schema_dict