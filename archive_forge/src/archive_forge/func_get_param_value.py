import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def get_param_value(params, key):
    """Looks up an expected parameter in a parsed params dict.

    Helper function, looks up an expected parameter in a parsed
    params dict and returns the result.  If params does not contain
    the requested key we raise an exception of the appropriate type.
    """
    try:
        return params[key]
    except KeyError:
        LOG.error('Request does not contain %s parameter!', key)
        raise exception.HeatMissingParameterError(key)