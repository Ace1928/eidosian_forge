from __future__ import absolute_import, division, print_function
import os
import re
from ast import literal_eval
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._json_compat import json
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import (
def check_required_by(requirements, parameters, options_context=None):
    """For each key in requirements, check the corresponding list to see if they
    exist in parameters.

    Accepts a single string or list of values for each key.

    :arg requirements: Dictionary of requirements
    :arg parameters: Dictionary of parameters
    :kwarg options_context: List of strings of parent key names if ``requirements`` are
        in a sub spec.

    :returns: Empty dictionary or raises :class:`TypeError` if the
    """
    result = {}
    if requirements is None:
        return result
    for key, value in requirements.items():
        if key not in parameters or parameters[key] is None:
            continue
        result[key] = []
        if isinstance(value, string_types):
            value = [value]
        for required in value:
            if required not in parameters or parameters[required] is None:
                result[key].append(required)
    if result:
        for key, missing in result.items():
            if len(missing) > 0:
                msg = "missing parameter(s) required by '%s': %s" % (key, ', '.join(missing))
                if options_context:
                    msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
                raise TypeError(to_native(msg))
    return result