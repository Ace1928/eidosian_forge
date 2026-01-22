from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def validate_cidr(cidr):
    """Validate CIDR. Return IP version of a CIDR string on success.

    :param cidr: Valid CIDR
    :type cidr: str
    :return: ``ipv4`` or ``ipv6``
    :rtype: str
    :raises ValueError: If ``cidr`` is not a valid CIDR
    """
    if CIDR_IPV4.match(cidr):
        return 'ipv4'
    elif CIDR_IPV6.match(cidr):
        return 'ipv6'
    raise ValueError('"{0}" is not a valid CIDR'.format(cidr))