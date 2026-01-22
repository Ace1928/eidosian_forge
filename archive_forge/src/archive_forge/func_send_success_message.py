from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import csv
import datetime
import enum
import os
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def send_success_message(task_status_queue, source_resource, destination_resource, md5_hash=None):
    """Send ManifestMessage for successful copy to central processing."""
    _send_manifest_message(task_status_queue, source_resource, destination_resource, ResultStatus.OK, md5_hash)