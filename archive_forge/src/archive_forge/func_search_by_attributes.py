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
def search_by_attributes(service, list_params=None, **kwargs):
    """
    Search for the entity by attributes. Nested entities don't support search
    via REST, so in case using search for nested entity we return all entities
    and filter them by specified attributes.
    """
    list_params = list_params or {}
    if 'search' in inspect.getfullargspec(service.list)[0]:
        res = service.list(search=' and '.join(('{0}="{1}"'.format(k, v) for k, v in kwargs.items())), **list_params)
    else:
        res = [e for e in service.list(**list_params) if len([k for k, v in kwargs.items() if getattr(e, k, None) == v]) == len(kwargs)]
    res = res or [None]
    return res[0]