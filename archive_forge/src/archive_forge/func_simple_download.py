from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import gcs_download
from googlecloudsdk.api_lib.storage.gcs_grpc import grpc_util
from googlecloudsdk.api_lib.storage.gcs_grpc import retry_util
@retry_util.grpc_default_retryer
def simple_download(self):
    """Downloads the object.

    On retriable errors, the entire download will be re-performed instead of
    resuming from a particular byte. This is useful for streaming download
    cases.

    Unlike Apitools, the GAPIC client doesn't retry the request by
    default, hence we are using the decorator.

    Returns:
      Encoding string for object if requested. Otherwise, None.
    """
    return self.launch()