from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def keyitem_to_dict(keyitem):
    return dict(kid=keyitem._id, version=keyitem.version, tags=keyitem._tags, manged=keyitem._managed, attributes=dict(enabled=keyitem.enabled, not_before=keyitem.not_before, expires=keyitem.expires_on, created=keyitem.created_on, updated=keyitem.updated_on, recovery_level=keyitem.recovery_level))