from pprint import pformat
from six import iteritems
import re
@automount_service_account_token.setter
def automount_service_account_token(self, automount_service_account_token):
    """
        Sets the automount_service_account_token of this V1PodSpec.
        AutomountServiceAccountToken indicates whether a service account token
        should be automatically mounted.

        :param automount_service_account_token: The
        automount_service_account_token of this V1PodSpec.
        :type: bool
        """
    self._automount_service_account_token = automount_service_account_token