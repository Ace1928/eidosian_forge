from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
Initializes task.

    Args:
      system_posix_data (SystemPosixData): Contains system-wide POSIX metadata.
      source_resource (resource_reference.ObjectResource): Contains custom POSIX
        metadata and URL for error logging.
      destination_resource (resource_reference.FileObjectResource): File to set
        POSIX metadata on.
      known_source_posix (PosixAttributes|None): Use pre-parsed POSIX data
        instead of extracting from source.
      known_destination_posix (PosixAttributes|None): Use pre-parsed POSIX data
        instead of extracting from destination.
    