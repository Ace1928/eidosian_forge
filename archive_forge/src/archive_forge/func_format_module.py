from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def format_module(self, item):
    if not item:
        return None
    format_item = dict(authentication=dict(), cloudToDeviceMessageCount=item.cloud_to_device_message_count, connectionState=item.connection_state, connectionStateUpdatedTime=item.connection_state_updated_time, deviceId=item.device_id, etag=item.etag, generationId=item.generation_id, lastActivityTime=item.last_activity_time, managedBy=item.managed_by, moduleId=item.module_id)
    if item.authentication:
        format_item['authentication']['symmetricKey'] = dict()
        format_item['authentication']['symmetricKey']['primaryKey'] = item.authentication.symmetric_key.primary_key
        format_item['authentication']['symmetricKey']['secondaryKey'] = item.authentication.symmetric_key.secondary_key
        format_item['authentication']['type'] = item.authentication.type
        format_item['authentication']['x509Thumbprint'] = dict()
        format_item['authentication']['x509Thumbprint']['primaryThumbprint'] = item.authentication.x509_thumbprint.primary_thumbprint
        format_item['authentication']['x509Thumbprint']['secondaryThumbprint'] = item.authentication.x509_thumbprint.secondary_thumbprint
    return format_item