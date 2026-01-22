from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def update_owners(self, group_id, client):
    current_owners = []
    if self.present_owners or self.absent_owners:
        current_owners = [object.object_id for object in list(client.groups.list_owners(group_id))]
    if self.present_owners:
        present_owners_by_object_id = self.dictionary_from_object_urls(self.present_owners)
        owners_to_add = list(set(present_owners_by_object_id.keys()) - set(current_owners))
        if owners_to_add:
            for owner_object_id in owners_to_add:
                client.groups.add_owner(group_id, present_owners_by_object_id[owner_object_id])
            self.results['changed'] = True
    if self.absent_owners:
        owners_to_remove = list(set(self.absent_owners).intersection(set(current_owners)))
        if owners_to_remove:
            for owner in owners_to_remove:
                client.groups.remove_owner(group_id, owner)
            self.results['changed'] = True