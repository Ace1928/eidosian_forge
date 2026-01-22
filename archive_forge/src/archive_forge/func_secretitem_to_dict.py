from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def secretitem_to_dict(secretitem):
    return dict(sid=secretitem._id, version=secretitem.version, tags=secretitem._tags, attributes=dict(enabled=secretitem._attributes.enabled, not_before=secretitem._attributes.not_before, expires=secretitem._attributes.expires, created=secretitem._attributes.created, updated=secretitem._attributes.updated, recovery_level=secretitem._attributes.recovery_level))