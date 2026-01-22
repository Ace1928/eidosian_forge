from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def raise_if_invalid_file_permissions(system_posix_data, resource, delete_path=None, known_posix=None):
    """Detects permissions causing inaccessibility.

  Can delete invalid file.

  Args:
    system_posix_data (SystemPosixData): Helps determine if file will be made
      inaccessible in local environment.
    resource (ObjectResource): Contains URL used for messages and custom POSIX
      metadata used to determine if setting invalid file permissions.
    delete_path (str|None): If present, will delete file before raising error.
      Useful if file has been downloaded and needs to be cleaned up.
    known_posix (PosixAttributes|None): Use pre-parsed POSIX data instead of
      extracting from source. Not super important here because the source is a
      cloud object and doesn't require an `os.stat` call to harvest metadata,
      but it would be strange if we used `known_posix` for callers and only
      `resource` here, especially if the values were different (which they
      shouldn't be). Be careful using this because, if the data is wrong, it
      could mess with these safety checks.

  Raises:
    SystemPermissionError: Has explanatory message about issue.
  """
    _, _, uid, gid, mode = known_posix or get_posix_attributes_from_cloud_resource(resource)
    if uid is gid is mode is None or platforms.OperatingSystem.IsWindows():
        return
    if os.geteuid() == 0:
        return
    import grp
    import pwd
    url_string = resource.storage_url.url_string
    if uid is not None:
        try:
            pwd.getpwuid(uid)
        except KeyError:
            error = errors.SystemPermissionError(_MISSING_UID_FORMAT.format(url_string, uid))
            _raise_error_and_maybe_delete_file(error, delete_path)
    if gid is not None:
        try:
            grp.getgrgid(gid)
        except (KeyError, OverflowError):
            error = errors.SystemPermissionError(_MISSING_GID_FORMAT.format(url_string, gid))
            _raise_error_and_maybe_delete_file(error, delete_path)
    if mode is None:
        mode_to_set = system_posix_data.default_mode
    else:
        mode_to_set = mode
    uid_to_set = uid or os.getuid()
    if uid is None or uid == os.getuid():
        if mode_to_set.base_ten_int & stat.S_IRUSR:
            return
        error = errors.SystemPermissionError(_INSUFFICIENT_USER_READ_ACCESS_FORMAT.format(url_string, uid_to_set, mode_to_set.base_eight_str))
        _raise_error_and_maybe_delete_file(error, delete_path)
    if gid is None or gid in system_posix_data.user_groups:
        if mode_to_set.base_ten_int & stat.S_IRGRP:
            return
        error = errors.SystemPermissionError(_INSUFFICIENT_GROUP_READ_ACCESS_FORMAT.format(url_string, '[user primary group]' if gid is None else gid, mode_to_set.base_eight_str))
        _raise_error_and_maybe_delete_file(error, delete_path)
    if mode_to_set.base_ten_int & stat.S_IROTH:
        return
    error = errors.SystemPermissionError(_INSUFFICIENT_OTHER_READ_ACCESS_FORMAT.format(url_string, uid_to_set, mode_to_set.base_eight_str))
    _raise_error_and_maybe_delete_file(error, delete_path)