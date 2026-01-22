from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_blob_tags(self, tags):
    if not self.check_mode:
        try:
            self.blob_service_client.get_blob_client(container=self.container, blob=self.blob).set_blob_metadata(metadata=tags)
        except Exception as exc:
            self.fail('Update blob tags {0}:{1} - {2}'.format(self.container, self.blob, str(exc)))
    self.blob_obj = self.get_blob()
    self.results['changed'] = True
    self.results['actions'].append('updated blob {0}:{1} tags.'.format(self.container, self.blob))
    self.results['container'] = self.container_obj
    self.results['blob'] = self.blob_obj