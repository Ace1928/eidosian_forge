from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
Reports back all tracked events via report method.

    Args:
      total_bytes (int): Amount of data transferred in bytes.
      time_delta (int): Time elapsed during the transfer in seconds.
      num_files (int): Number of files processed
    