from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_service_principal_profile_dict(serviceprincipalprofile):
    """
    Helper method to deserialize a ContainerServiceServicePrincipalProfile to a dict
    Note: For security reason, the service principal secret is skipped on purpose.
    :param: serviceprincipalprofile: ContainerServiceServicePrincipalProfile with the Azure callback object
    :return: dict with the state on Azure
    """
    return dict(client_id=serviceprincipalprofile.client_id)