import re
import sys
def get_list_opt(options, optname, default=None):
    val = options.get(optname, default)
    if isinstance(val, string_types):
        return val.split()
    elif isinstance(val, (list, tuple)):
        return list(val)
    else:
        raise OptionError('Invalid type %r for option %s; you must give a list value' % (val, optname))