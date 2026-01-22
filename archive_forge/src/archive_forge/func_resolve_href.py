from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def resolve_href(value):
    try:
        value = connection.follow_link(value)
    except sdk.Error:
        value = None
    nested_obj = dict(((attr, convert_value(getattr(value, attr))) for attr in attributes if getattr(value, attr, None) is not None))
    nested_obj['id'] = getattr(value, 'id', None)
    nested_obj['href'] = getattr(value, 'href', None)
    return nested_obj