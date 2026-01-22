from pprint import pformat
from six import iteritems
import re
@secret_name.setter
def secret_name(self, secret_name):
    """
        Sets the secret_name of this V1AzureFileVolumeSource.
        the name of secret that contains Azure Storage Account Name and Key

        :param secret_name: The secret_name of this V1AzureFileVolumeSource.
        :type: str
        """
    if secret_name is None:
        raise ValueError('Invalid value for `secret_name`, must not be `None`')
    self._secret_name = secret_name