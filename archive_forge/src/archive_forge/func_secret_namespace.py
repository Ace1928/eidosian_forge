from pprint import pformat
from six import iteritems
import re
@secret_namespace.setter
def secret_namespace(self, secret_namespace):
    """
        Sets the secret_namespace of this V1AzureFilePersistentVolumeSource.
        the namespace of the secret that contains Azure Storage Account Name and
        Key default is the same as the Pod

        :param secret_namespace: The secret_namespace of this
        V1AzureFilePersistentVolumeSource.
        :type: str
        """
    self._secret_namespace = secret_namespace