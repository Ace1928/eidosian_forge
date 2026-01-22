from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def normalize_ipam_config_key(key):
    """Normalizes IPAM config keys returned by Docker API to match Ansible keys.

    :param key: Docker API key
    :type key: str
    :return Ansible module key
    :rtype str
    """
    special_cases = {'AuxiliaryAddresses': 'aux_addresses'}
    return special_cases.get(key, key.lower())