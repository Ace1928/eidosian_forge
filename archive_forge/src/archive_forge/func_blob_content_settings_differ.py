from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def blob_content_settings_differ(self):
    if self.content_type or self.content_encoding or self.content_language or self.content_disposition or self.cache_control or self.content_md5:
        settings = dict(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=self.content_md5)
        if self.blob_obj['content_settings'] != settings:
            return True
    return False