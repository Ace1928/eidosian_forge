from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import random
import re
import shlex
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleActionSkip, AnsibleActionFail, AnsibleAuthenticationFailure
from ansible.executor.module_common import modify_module
from ansible.executor.interpreter_discovery import discover_interpreter, InterpreterDiscoveryRequiredError
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.module_utils.errors import UnsupportedError
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.jsonify import jsonify
from ansible.release import __version__
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var, AnsibleUnsafeText
from ansible.vars.clean import remove_internal_keys
from ansible.utils.plugin_docs import get_versioned_doclink
def validate_argument_spec(self, argument_spec=None, mutually_exclusive=None, required_together=None, required_one_of=None, required_if=None, required_by=None):
    """Validate an argument spec against the task args

        This will return a tuple of (ValidationResult, dict) where the dict
        is the validated, coerced, and normalized task args.

        Be cautious when directly passing ``new_module_args`` directly to a
        module invocation, as it will contain the defaults, and not only
        the args supplied from the task. If you do this, the module
        should not define ``mututally_exclusive`` or similar.

        This code is roughly copied from the ``validate_argument_spec``
        action plugin for use by other action plugins.
        """
    new_module_args = self._task.args.copy()
    validator = ArgumentSpecValidator(argument_spec, mutually_exclusive=mutually_exclusive, required_together=required_together, required_one_of=required_one_of, required_if=required_if, required_by=required_by)
    validation_result = validator.validate(new_module_args)
    new_module_args.update(validation_result.validated_parameters)
    try:
        error = validation_result.errors[0]
    except IndexError:
        error = None
    if error:
        msg = validation_result.errors.msg
        if isinstance(error, UnsupportedError):
            msg = f'Unsupported parameters for ({self._load_name}) module: {msg}'
        raise AnsibleActionFail(msg)
    return (validation_result, new_module_args)