from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
Generate the YAML manifest to deploy the Connect Agent.

    Args:
      option: an instance of ConnectAgentOption.

    Returns:
      A slice of connect agent manifest resources.
    Raises:
      Error: if the API call to generate connect agent manifest failed.
    