from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import patch_file_posix_task
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.objects import patch_object_task
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def get_existing_or_placeholder_destination_resource(path, ignore_symlinks=True):
    """Returns existing valid container or UnknownResource or raises."""
    resource_iterator = wildcard_iterator.get_wildcard_iterator(path, fields_scope=cloud_api.FieldsScope.SHORT, get_bucket_metadata=True, ignore_symlinks=ignore_symlinks)
    plurality_checkable_resource_iterator = plurality_checkable_iterator.PluralityCheckableIterator(resource_iterator)
    if plurality_checkable_resource_iterator.is_empty():
        if wildcard_iterator.contains_wildcard(path):
            raise errors.InvalidUrlError('Wildcard pattern matched nothing. ' + _NO_MATCHES_MESSAGE.format(path))
        return resource_reference.UnknownResource(storage_url.storage_url_from_string(path))
    if plurality_checkable_resource_iterator.is_plural():
        raise errors.InvalidUrlError('{} matched more than one URL: {}'.format(path, list(plurality_checkable_resource_iterator)))
    resource = list(plurality_checkable_resource_iterator)[0]
    if resource.is_container():
        return resource
    raise errors.InvalidUrlError('{} matched non-container URL: {}'.format(path, resource))