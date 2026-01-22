from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.warnings import deprecate, warn
from ansible.module_utils.common.validation import (
from ansible.module_utils.errors import (
@property
def unsupported_parameters(self):
    """:class:`set` of unsupported parameter names."""
    return self._unsupported_parameters