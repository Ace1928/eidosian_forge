from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import streaming_download_task
def get_cat_task_iterator(source_iterator, show_url, start_byte, end_byte):
    """An iterator that yields StreamingDownloadTasks for cat sources.

  Given a list of strings that are object URLs ("gs://foo/object1"), yield a
  StreamingDownloadTask.

  Args:
    source_iterator (NameExpansionIterator): Yields sources resources that
      should be packaged in StreamingDownloadTasks.
    show_url (bool): Says whether or not to print the header before each
      object's content.
    start_byte (int): The byte index to start streaming from.
    end_byte (int|None): The byte index to stop streaming from.

  Yields:
    StreamingDownloadTask

  """
    stdout = os.fdopen(1, 'wb')
    dummy_destination_resource = resource_reference.FileObjectResource(storage_url.FileUrl('-'))
    for item in source_iterator:
        yield streaming_download_task.StreamingDownloadTask(item.resource, dummy_destination_resource, download_stream=stdout, show_url=show_url, start_byte=_get_start_byte(start_byte, item.resource.size), end_byte=end_byte)