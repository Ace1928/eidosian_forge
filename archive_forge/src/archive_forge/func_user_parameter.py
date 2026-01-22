import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
def user_parameter(schema_item):
    name, schema = schema_item
    return Parameter(name, schema, user_params.get(name))