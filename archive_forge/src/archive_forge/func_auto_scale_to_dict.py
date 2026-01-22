from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def auto_scale_to_dict(instance):
    if not instance:
        return dict()
    return dict(id=to_native(instance.id or ''), name=to_native(instance.name), location=to_native(instance.location), profiles=[profile_to_dict(p) for p in instance.profiles or []], notifications=[notification_to_dict(n) for n in instance.notifications or []], enabled=instance.enabled, target=to_native(instance.target_resource_uri), tags=instance.tags)