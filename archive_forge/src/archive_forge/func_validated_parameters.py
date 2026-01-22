from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.warnings import deprecate, warn
from ansible.module_utils.common.validation import (
from ansible.module_utils.errors import (
@property
def validated_parameters(self):
    """Validated and coerced parameters."""
    return self._validated_parameters