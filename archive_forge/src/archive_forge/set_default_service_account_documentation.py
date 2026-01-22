from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
Set the default service account on the project.

    *{command}* is used to configure the default service account on project.

  The project's default service account is used when a new instance is
  created unless a custom service account is set via --scopes or
  --no-scopes. Existing instances are not affected.

  For example,

    $ {command} --service-account=example@developers.gserviceaccount.com
    $ gcloud compute instances create instance-name

  will set the project's default service account as
  example@developers.gserviceaccount.com. The instance created will have
  example@developers.gserviceaccount.com as the service account associated
  with because no service account email was specified in the
  "instances create" command.

  To remove the default service account from the project, issue the command:

    $ gcloud compute project-info set-default-service-account \
        --no-service-account

  The required permission to execute this command is
  `compute.projects.setDefaultServiceAccount`. If needed, you can include this
  permission, or choose any of the following preexisting IAM roles that contain
  this particular permission:

    * Owner
    * Editor
    * Compute Admin
  