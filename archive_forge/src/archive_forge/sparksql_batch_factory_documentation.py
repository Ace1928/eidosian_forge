from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
Uploads local files and creates a SparkSqlBatch message.

    Uploads user local files and change the URIs to local files to uploaded
    URIs.
    Creates a SparkSqlBatch message.

    Args:
      args: Parsed arguments.

    Returns:
      A SparkSqlBatch message instance.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    