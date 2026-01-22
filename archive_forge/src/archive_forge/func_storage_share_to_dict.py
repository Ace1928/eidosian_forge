from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def storage_share_to_dict(self, storage_share):
    """
        Transform Azure RM Storage share object to dictionary
        :param storage_share: contains information about storage file share
        :type storage_share: FileShare
        :return: dict generated from storage_share
        """
    return dict(id=storage_share.id, name=storage_share.name, type=storage_share.type, etag=storage_share.etag.replace('"', ''), last_modified_time=storage_share.last_modified_time, metadata=storage_share.metadata, share_quota=storage_share.share_quota, access_tier=storage_share.access_tier, access_tier_change_time=storage_share.access_tier_change_time, enabled_protocols=storage_share.enabled_protocols, root_squash=storage_share.root_squash, version=storage_share.version, deleted=storage_share.deleted, deleted_time=storage_share.deleted_time, remaining_retention_days=storage_share.remaining_retention_days, access_tier_status=storage_share.access_tier_status, share_usage_bytes=storage_share.share_usage_bytes) if storage_share else None