from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def deletedkeyitem_to_dict(keyitem):
    item = delete_item_to_dict(keyitem)
    item['recovery_id'] = keyitem.recovery_id
    item['scheduled_purge_date'] = keyitem.scheduled_purge_date
    item['deleted_date'] = keyitem.deleted_date
    return item