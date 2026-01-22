from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def faildeploy(self, param):
    """
        Helper method to push fail message in the console.
        Useful to notify that the users cannot change some values in a Availability Set

        :param: variable's name impacted
        :return: void
        """
    self.fail('You tried to change {0} but is was unsuccessful. An Availability Set is immutable, except tags'.format(str(param)))