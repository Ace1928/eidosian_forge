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
def check_missing_parameters(parameters, required_parameters=None):
    """This is for checking for required params when we can not check via
    argspec because we need more information than is simply given in the argspec.

    Raises :class:`TypeError` if any required parameters are missing

    :arg parameters: Dictionary of parameters
    :arg required_parameters: List of parameters to look for in the given parameters.

    :returns: Empty list or raises :class:`TypeError` if the check fails.
    """
    missing_params = []
    if required_parameters is None:
        return missing_params
    for param in required_parameters:
        if not parameters.get(param):
            missing_params.append(param)
    if missing_params:
        msg = 'missing required arguments: %s' % ', '.join(missing_params)
        raise TypeError(to_native(msg))
    return missing_params