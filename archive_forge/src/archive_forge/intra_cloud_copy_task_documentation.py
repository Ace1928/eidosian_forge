from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
Initializes task.

    Args:
      source_resource (resource_reference.Resource): Must contain the full
        object path. Directories will not be accepted.
      destination_resource (resource_reference.Resource): Must contain the full
        object path. Directories will not be accepted. Existing objects at the
        this location will be overwritten.
      delete_source (bool): If copy completes successfully, delete the source
        object afterwards.
      fetch_source_fields_scope (FieldsScope|None): If present, refetch
        source_resource, populated with metadata determined by this FieldsScope.
        Useful for lazy or parallelized GET calls.
      posix_to_set (PosixAttributes|None): See parent class.
      print_created_message (bool): See parent class.
      print_source_version (bool): See parent class.
      user_request_args (UserRequestArgs|None): See parent class
      verbose (bool): See parent class.
    