from __future__ import (absolute_import, division, print_function)
import os
import re
from collections.abc import MutableMapping, MutableSequence
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.plugins.loader import connection_loader
from ansible.utils.display import Display
def strip_internal_keys(dirty, exceptions=None):
    if exceptions is None:
        exceptions = tuple()
    if isinstance(dirty, MutableSequence):
        for element in dirty:
            if isinstance(element, (MutableMapping, MutableSequence)):
                strip_internal_keys(element, exceptions=exceptions)
    elif isinstance(dirty, MutableMapping):
        for k in list(dirty.keys()):
            if isinstance(k, six.string_types):
                if k.startswith('_ansible_') and k not in exceptions:
                    del dirty[k]
                    continue
            if isinstance(dirty[k], (MutableMapping, MutableSequence)):
                strip_internal_keys(dirty[k], exceptions=exceptions)
    else:
        raise AnsibleError('Cannot strip invalid keys from %s' % type(dirty))
    return dirty