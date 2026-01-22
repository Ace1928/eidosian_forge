from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_container_tags(self, tags):
    if not self.check_mode:
        try:
            self.blob_service_client.get_container_client(container=self.container).set_container_metadata(metadata=tags)
        except Exception as exc:
            self.fail('Error updating container tags {0} - {1}'.format(self.container, str(exc)))
    self.container_obj = self.get_container()
    self.results['changed'] = True
    self.results['actions'].append('updated container {0} tags.'.format(self.container))
    self.results['container'] = self.container_obj