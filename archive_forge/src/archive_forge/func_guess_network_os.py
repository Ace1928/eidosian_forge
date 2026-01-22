from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.errors import AnsibleError
from ansible.plugins import AnsiblePlugin
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
@staticmethod
def guess_network_os(obj):
    """
        Identifies the operating system of network device.
        :param obj: ncclient manager connection instance
        :return: The name of network operating system.
        """
    pass