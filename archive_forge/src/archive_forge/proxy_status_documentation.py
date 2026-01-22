from googlecloudsdk.api_lib.container.fleet import debug_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.mesh import istioctl_backend
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Retrive the envoy configuration sync status or the detailed config dump.
  