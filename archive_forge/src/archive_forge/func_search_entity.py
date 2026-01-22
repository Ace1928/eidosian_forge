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
def search_entity(self, search_params=None, list_params=None):
    """
        Always first try to search by `ID`, if ID isn't specified,
        check if user constructed special search in `search_params`,
        if not search by `name`.
        """
    entity = None
    if 'id' in self._module.params and self._module.params['id'] is not None:
        entity = get_entity(self._service.service(self._module.params['id']), get_params=list_params)
    elif search_params is not None:
        entity = search_by_attributes(self._service, list_params=list_params, **search_params)
    elif self._module.params.get('name') is not None:
        entity = search_by_attributes(self._service, list_params=list_params, name=self._module.params['name'])
    return entity