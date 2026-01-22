import collections
import functools
import inspect
import re
import netaddr
from os_ken.lib.packet import ether_types
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from webob import exc
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.plugins import directory
from neutron_lib.services.qos import constants as qos_consts
def validate_values(data, valid_values=None, valid_values_display=None):
    """Validate that the provided 'data' is within 'valid_values'.

    :param data: The data to check within valid_values.
    :param valid_values: A collection of values that 'data' must be in to be
        valid. The collection can be any type that supports the 'in' operation.
    :param valid_values_display: A string to display that describes the valid
        values. This string is only displayed when an invalid value is
        encountered.
        If no string is provided, the string "valid_values" will be used.
    :returns: The message to return if data not in valid_values.
    :raises: TypeError if the values for 'data' or 'valid_values' are not
        compatible for comparison or doesn't have __contains__.
        If TypeError is raised this is considered a programming error and the
        inputs (data) and (valid_values) must be checked so this is never
        raised on validation.
    """
    if valid_values is None:
        return
    contains = getattr(valid_values, '__contains__', None)
    if callable(contains):
        try:
            if data not in valid_values:
                valid_values_display = valid_values_display or 'valid_values'
                msg_data = {'data': data, 'valid_values': valid_values_display}
                msg = '%(data)s is not in %(valid_values)s'
                LOG.debug(msg, msg_data)
                return _(msg) % msg_data
        except TypeError as e:
            msg = _("'data' of type '%(typedata)s' and 'valid_values' of type '%(typevalues)s' are not compatible for comparison") % {'typedata': type(data), 'typevalues': type(valid_values)}
            raise TypeError(msg) from e
    else:
        msg = _("'valid_values' does not support membership operations")
        raise TypeError(msg)