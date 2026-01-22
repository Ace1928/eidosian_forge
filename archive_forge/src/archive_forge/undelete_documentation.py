from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core import log
Undelete a root Certificate Authority.

    Restores a root Certificate Authority that has been deleted. A Certificate
    Authority can be undeleted within 30 days of being deleted. Use this command
    to halt the deletion process. An undeleted CA will move to DISABLED state.

    ## EXAMPLES

    To undelete a root CA:

        $ {command} prod-root --location=us-west1 --pool=my-pool
  