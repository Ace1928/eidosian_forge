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
def ovirt_info_full_argument_spec(**kwargs):
    """
    Extend parameters of info module with parameters which are common to all
    oVirt info modules.

    :param kwargs: kwargs to be extended
    :return: extended dictionary with common parameters
    """
    spec = dict(auth=__get_auth_dict(), fetch_nested=dict(default=False, type='bool'), nested_attributes=dict(type='list', default=list(), elements='str'), follow=dict(default=list(), type='list', elements='str', aliases=['follows']))
    spec.update(kwargs)
    return spec