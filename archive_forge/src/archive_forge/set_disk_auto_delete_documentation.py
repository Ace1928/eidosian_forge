from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.console import console_io
Gets an attached disk with the specified device name.

    Args:
      resources: resources.Registry, The resource registry
      device_name: str, device name of the attached disk.
      instance_ref: Reference of the instance instance.
      instance: Instance object.

    Returns:
      An attached disk object.

    Raises:
      compute_exceptions.ArgumentError: If a disk with device name cannot be
          found attached to the instance.
    