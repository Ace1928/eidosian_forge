from pprint import pformat
from six import iteritems
import re
@value_from.setter
def value_from(self, value_from):
    """
        Sets the value_from of this V1EnvVar.
        Source for the environment variable's value. Cannot be used if value is
        not empty.

        :param value_from: The value_from of this V1EnvVar.
        :type: V1EnvVarSource
        """
    self._value_from = value_from