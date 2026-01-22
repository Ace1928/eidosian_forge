from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import resource_args
from googlecloudsdk.core.resource import resource_projector
List the instance details for an OS patch job.

  ## EXAMPLES

  To list the instance details for each instance in the patch job `job1`, run:

        $ {command} job1

  