from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.command_lib.projects import util as projects_util
def ensure_pubsub_sa_has_token_creator_role():
    """Ensures the project's Pub/Sub service account has permission to create tokens.

  If the permission is missing, prompts the user to grant it. If the console
  cannot prompt, prints a warning instead.
  """
    pubsub_sa = 'service-{}@gcp-sa-pubsub.iam.gserviceaccount.com'.format(projects_util.GetProjectNumber(api_util.GetProject()))
    api_util.PromptToBindRoleIfMissing(pubsub_sa, 'roles/iam.serviceAccountTokenCreator', alt_roles=['roles/pubsub.serviceAgent'], reason='Pub/Sub needs this role to create identity tokens. For more details, please see https://cloud.google.com/pubsub/docs/push#authentication')