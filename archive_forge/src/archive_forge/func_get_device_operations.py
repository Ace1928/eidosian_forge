from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.errors import AnsibleError
from ansible.plugins import AnsiblePlugin
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def get_device_operations(self, server_capabilities):
    """
        Retrieve remote host capability from Netconf server hello message.
        :param server_capabilities: Server capabilities received during Netconf session initialization
        :return: Remote host capabilities in dictionary format
        """
    operations = {}
    capabilities = '\n'.join(server_capabilities)
    operations['supports_commit'] = ':candidate' in capabilities
    operations['supports_defaults'] = ':with-defaults' in capabilities
    operations['supports_confirm_commit'] = ':confirmed-commit' in capabilities
    operations['supports_startup'] = ':startup' in capabilities
    operations['supports_xpath'] = ':xpath' in capabilities
    operations['supports_writable_running'] = ':writable-running' in capabilities
    operations['supports_validate'] = ':validate' in capabilities
    operations['lock_datastore'] = []
    if operations['supports_writable_running']:
        operations['lock_datastore'].append('running')
    if operations['supports_commit']:
        operations['lock_datastore'].append('candidate')
    if operations['supports_startup']:
        operations['lock_datastore'].append('startup')
    operations['supports_lock'] = bool(operations['lock_datastore'])
    return operations