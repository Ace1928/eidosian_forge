from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
Uploads local files and creates a SparkBatch message.

    Uploads user local files and change the URIs to local files to point to
    uploaded URIs.
    Creates a SparkBatch message from parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      SparkBatch: A SparkBatch message.

    Raises:
      AttributeError: Main class and jar are missing, or both were provided.
      Bucket is required to upload local files, but not specified.
    