from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.api_lib.storage.gcs_grpc import client as gcs_grpc_client
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.storage.gcs_xml import client as gcs_xml_client
from googlecloudsdk.api_lib.storage.s3_xml import client as s3_xml_client
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def get_capabilities(provider):
    """Gets the capabilities of a cloud provider.

  Args:
    provider (storage_url.ProviderPrefix): Cloud provider prefix.

  Returns:
    The CloudApi.capabilities attribute for the requested provider.

  Raises:
    Error: If provider is not a cloud scheme in storage_url.ProviderPrefix.
  """
    api_class = _get_api_class(provider)
    return api_class.capabilities