import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def validate_dns_domain(data, max_len=db_constants.FQDN_FIELD_SIZE):
    """Validate DNS domain.

    :param data: The data to validate.
    :param max_len: An optional cap on the length of the string.
    :returns: None if data is valid, otherwise a human readable message
        indicating why validation failed.
    """
    msg = validators.validate_string(data)
    if msg:
        return msg
    if not data:
        return
    if not data.endswith('.'):
        msg = _("'%s' is not a FQDN") % data
        return msg
    msg = _validate_dns_format(data, max_len)
    if msg:
        return msg
    length = len(data)
    if length > max_len - 2:
        msg = _("'%(data)s' contains %(length)s characters. Adding a sub-domain will cause it to exceed the maximum length of a FQDN of '%(max_len)s'") % {'data': data, 'length': length, 'max_len': max_len}
        return msg