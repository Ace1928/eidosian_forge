from pprint import pformat
from six import iteritems
import re
@secret_key_ref.setter
def secret_key_ref(self, secret_key_ref):
    """
        Sets the secret_key_ref of this V1EnvVarSource.
        Selects a key of a secret in the pod's namespace

        :param secret_key_ref: The secret_key_ref of this V1EnvVarSource.
        :type: V1SecretKeySelector
        """
    self._secret_key_ref = secret_key_ref