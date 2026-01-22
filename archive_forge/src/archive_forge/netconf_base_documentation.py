from __future__ import absolute_import, division, print_function
from abc import abstractmethod
from functools import wraps
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.plugins.netconf import NetconfBase as NetconfBaseBase

        Retrieve remote host capability from Netconf server hello message.
        :param server_capabilities: Server capabilities received during Netconf session initialization
        :return: Remote host capabilities in dictionary format
        