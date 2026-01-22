from pprint import pformat
from six import iteritems
import re
@served.setter
def served(self, served):
    """
        Sets the served of this V1beta1CustomResourceDefinitionVersion.
        Served is a flag enabling/disabling this version from being served via
        REST APIs

        :param served: The served of this
        V1beta1CustomResourceDefinitionVersion.
        :type: bool
        """
    if served is None:
        raise ValueError('Invalid value for `served`, must not be `None`')
    self._served = served