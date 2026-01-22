from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
def raise_error_if_not_bucket(command_list, url):
    if not (isinstance(url, storage_url.CloudUrl) and url.is_bucket()):
        _raise_error_for_wrong_resource_type(command_list, 'bucket', 'gs://bucket', url)