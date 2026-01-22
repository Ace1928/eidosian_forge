from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
Sends a ListAvailableFeatures request and returns the features.

    Args:
      project: String representing the project to use for the request.
      region: The region to use. If not set, the global scope is used.

    Returns:
      List of strings representing the list of available features.
    