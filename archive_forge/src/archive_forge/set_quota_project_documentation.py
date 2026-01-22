from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.resource_manager import completers
Update or add a quota project in application default credentials (ADC).

  A quota project is a Google Cloud Project that will be used for billing
  and quota limits.

  Before running this command, an ADC must already be generated using
  $ gcloud auth application-default login.
  The quota project defined in the ADC will be used by the Google client
  libraries.
  The existing application default credentials must have the
  `serviceusage.services.use` permission on the given project.

  ## EXAMPLES

  To update the quota project in application default credentials to
  `my-quota-project`, run:

    $ {command} my-quota-project
  