from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_azuread_config(auth_method, msg):
    """Provision FeatureSpec AzureADConfig from the parsed yaml file.

  Args:
    auth_method: YamlConfigFile, The data loaded from the yaml file given by the
      user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing a single
    Azure AD auth method for the IdentityServiceFeatureSpec.
  """
    if 'name' not in auth_method:
        raise exceptions.Error('AzureAD Authentication method must contain name.')
    auth_method_proto = msg.IdentityServiceAuthMethod()
    auth_method_proto.name = auth_method['name']
    auth_method_proto.azureadConfig = msg.IdentityServiceAzureADConfig()
    if 'proxy' in auth_method:
        auth_method_proto.proxy = auth_method['proxy']
    azuread_config = auth_method['azureAD']
    if 'clientID' not in azuread_config or 'kubectlRedirectURI' not in azuread_config or 'tenant' not in azuread_config:
        err_msg = 'Authentication method [{}] must contain clientID, kubectlRedirectURI, and tenant.'.format(auth_method['name'])
        raise exceptions.Error(err_msg)
    auth_method_proto.azureadConfig.clientId = azuread_config['clientID']
    auth_method_proto.azureadConfig.kubectlRedirectUri = azuread_config['kubectlRedirectURI']
    auth_method_proto.azureadConfig.tenant = azuread_config['tenant']
    if 'clientSecret' in azuread_config:
        auth_method_proto.azureadConfig.clientSecret = azuread_config['clientSecret']
    if 'userClaim' in azuread_config:
        auth_method_proto.azureadConfig.userClaim = azuread_config['userClaim']
    if 'groupFormat' in azuread_config:
        auth_method_proto.azureadConfig.groupFormat = azuread_config['groupFormat']
    return auth_method_proto