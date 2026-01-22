from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
@property
def literals(self):
    """Mutable dict-like object for env vars with a string literal.

    Note that if neither value nor valueFrom is specified, the list entry will
    be treated as a literal empty string.

    Returns:
      A mutable, dict-like object for managing string literal env vars.
    """
    return k8s_object.KeyValueListAsDictionaryWrapper(self._env_vars, self._env_var_class, filter_func=lambda env_var: env_var.valueFrom is None)