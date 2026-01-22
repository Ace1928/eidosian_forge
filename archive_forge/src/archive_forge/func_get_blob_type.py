from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_blob_type(self, blob_type):
    if blob_type == 'block':
        return BlobType.BlockBlob
    elif blob_type == 'page':
        return BlobType.PageBlob
    else:
        return BlobType.AppendBlob