from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
Issues the request necessary for uploading a route policy into a Router.

    Args:
      args: contains arguments passed to the command.

    Returns:
      The result of patching the router uploading the route policy.
    