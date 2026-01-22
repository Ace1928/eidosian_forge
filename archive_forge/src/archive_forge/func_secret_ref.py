from pprint import pformat
from six import iteritems
import re
@secret_ref.setter
def secret_ref(self, secret_ref):
    """
        Sets the secret_ref of this V1CinderPersistentVolumeSource.
        Optional: points to a secret object containing parameters used to
        connect to OpenStack.

        :param secret_ref: The secret_ref of this
        V1CinderPersistentVolumeSource.
        :type: V1SecretReference
        """
    self._secret_ref = secret_ref