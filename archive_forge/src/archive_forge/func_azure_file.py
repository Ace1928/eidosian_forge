from pprint import pformat
from six import iteritems
import re
@azure_file.setter
def azure_file(self, azure_file):
    """
        Sets the azure_file of this V1PersistentVolumeSpec.
        AzureFile represents an Azure File Service mount on the host and bind
        mount to the pod.

        :param azure_file: The azure_file of this V1PersistentVolumeSpec.
        :type: V1AzureFilePersistentVolumeSource
        """
    self._azure_file = azure_file