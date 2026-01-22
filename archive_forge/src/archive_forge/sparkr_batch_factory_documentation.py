from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
Upload local files and creates a SparkRBatch message.

    Upload user local files and change local file URIs to point to the uploaded
    URIs.
    Creates a SparkRBatch message based on parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      A SparkRBatch message.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    