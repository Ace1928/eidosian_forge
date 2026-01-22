from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.action import ActionBase
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.vars import combine_vars

        Validate an argument specification against a provided set of data.

        The `validate_argument_spec` module expects to receive the arguments:
            - argument_spec: A dict whose keys are the valid argument names, and
                  whose values are dicts of the argument attributes (type, etc).
            - provided_arguments: A dict whose keys are the argument names, and
                  whose values are the argument value.

        :param tmp: Deprecated. Do not use.
        :param task_vars: A dict of task variables.
        :return: An action result dict, including a 'argument_errors' key with a
            list of validation errors found.
        