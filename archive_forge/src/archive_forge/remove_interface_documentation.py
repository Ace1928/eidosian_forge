from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking.routers import routers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.networking import resource_args
from googlecloudsdk.core import log
remove an interface on a Distributed Cloud Edge Network router.

  *{command}* is used to remove an interface to a Distributed Cloud Edge
  Network router.
  