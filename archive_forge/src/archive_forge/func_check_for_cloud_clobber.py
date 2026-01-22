from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def check_for_cloud_clobber(user_request_args, api_client, destination_resource):
    """Returns if cloud destination object exists if no-clobber enabled."""
    if not (user_request_args and user_request_args.no_clobber):
        return False
    try:
        api_client.get_object_metadata(destination_resource.storage_url.bucket_name, destination_resource.storage_url.object_name, fields_scope=cloud_api.FieldsScope.SHORT)
    except api_errors.NotFoundError:
        return False
    return True