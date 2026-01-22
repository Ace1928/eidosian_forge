import logging
from oslo_utils import strutils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient import exceptions
def extract_extra_specs(extra_specs, specs_to_add, bool_specs=constants.BOOL_SPECS):
    try:
        for item in specs_to_add:
            key, value = item.split('=', 1)
            if key in extra_specs:
                msg = "Argument '%s' value specified twice." % key
                raise exceptions.CommandError(msg)
            elif key in bool_specs:
                if strutils.is_valid_boolstr(value):
                    extra_specs[key] = value.capitalize()
                else:
                    msg = "Argument '%s' is of boolean type and has invalid value: %s" % (key, str(value))
                    raise exceptions.CommandError(msg)
            else:
                extra_specs[key] = value
    except ValueError:
        msg = LOG.error(_('Wrong format: specs should be key=value pairs.'))
        raise exceptions.CommandError(msg)
    return extra_specs