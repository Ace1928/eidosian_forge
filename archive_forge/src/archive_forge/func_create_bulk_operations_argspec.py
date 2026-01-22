from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.argspec import (
def create_bulk_operations_argspec(provider_information):
    """
    If the provider supports bulk operations, return an ArgumentSpec object with appropriate
    options. Otherwise return an empty one.
    """
    if not provider_information.supports_bulk_actions():
        return ArgumentSpec()
    return ArgumentSpec(argument_spec=dict(bulk_operation_threshold=dict(type='int', default=2)))