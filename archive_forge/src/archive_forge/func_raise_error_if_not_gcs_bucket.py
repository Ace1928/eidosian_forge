from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
def raise_error_if_not_gcs_bucket(command_list, url):
    raise_error_if_not_gcs(command_list, url)
    raise_error_if_not_bucket(command_list, url)