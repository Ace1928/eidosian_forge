from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
def resolve_for_command(self, command_name, config_key, default_value=None):
    """
    Check for a per-command value or override of the given configuration key
    and return it if it exists. Otherwise return the global configuration value
    for that key.
    """
    configpath = config_key.split('.')
    fieldname = configpath.pop(-1)
    configobj = self
    for subname in configpath:
        nextobj = getattr(configobj, subname, None)
        if nextobj is None:
            raise ValueError('Config object {} does not have a subobject named {}'.format(type(configobj).__name__, subname))
        configobj = nextobj
    if hasattr(configobj, fieldname):
        assert default_value is None, 'Specifying a default value is not allowed if the config key exists in the global configuration ({})'.format(config_key)
        default_value = getattr(configobj, fieldname)
    command_dict = self.misc.per_command_.get(command_name.lower(), {})
    if config_key in command_dict:
        return command_dict[config_key]
    if fieldname in command_dict:
        return command_dict[fieldname]
    return default_value