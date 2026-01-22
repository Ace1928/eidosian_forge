import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
@staticmethod
def schema_from_params(params_snippet):
    """Create properties schema from the parameters section of a template.

        :param params_snippet: parameter definition from a template
        :returns: equivalent properties schemata for the specified parameters
        """
    if params_snippet:
        return dict(((n, Schema.from_parameter(p)) for n, p in params_snippet.items()))
    return {}