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
def parse_csv_line_to_resource(line, is_managed_folder=False):
    """Parses a line from files listing of rsync source and destination.

  Args:
    line (str|None): CSV line. See `get_csv_line_from_resource` docstring.
    is_managed_folder (bool): If True, returns a managed folder resource for
      cloud URLs. Otherwise, returns an object URL.

  Returns:
    FileObjectResource|ManagedFolderResource|ObjectResource|None: Resource
      containing data needed for rsync if data line given.
  """
    if not line:
        return None
    line_information = line.rstrip().rsplit(',', _CSV_COLUMNS_COUNT)
    url_string = line_information[0]
    url_object = storage_url.storage_url_from_string(url_string)
    if isinstance(url_object, storage_url.FileUrl):
        return resource_reference.FileObjectResource(url_object)
    if is_managed_folder:
        return resource_reference.ManagedFolderResource(url_object)
    _, etag_string, size_string, storage_class_string, atime_string, mtime_string, uid_string, gid_string, mode_base_eight_string, crc32c_string, md5_string = line.rstrip().rsplit(',', _CSV_COLUMNS_COUNT)
    cloud_object = resource_reference.ObjectResource(url_object, etag=etag_string if etag_string else None, size=int(size_string) if size_string else None, storage_class=storage_class_string if storage_class_string else None, crc32c_hash=crc32c_string if crc32c_string else None, md5_hash=md5_string if md5_string else None, custom_fields={})
    posix_util.update_custom_metadata_dict_with_posix_attributes(cloud_object.custom_fields, posix_util.PosixAttributes(atime=int(atime_string) if atime_string else None, mtime=int(mtime_string) if mtime_string else None, uid=int(uid_string) if uid_string else None, gid=int(gid_string) if gid_string else None, mode=posix_util.PosixMode.from_base_eight_str(mode_base_eight_string) if mode_base_eight_string else None))
    return cloud_object