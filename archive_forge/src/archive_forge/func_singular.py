from pprint import pformat
from six import iteritems
import re
@singular.setter
def singular(self, singular):
    """
        Sets the singular of this V1beta1CustomResourceDefinitionNames.
        Singular is the singular name of the resource.  It must be all lowercase
        Defaults to lowercased <kind>

        :param singular: The singular of this
        V1beta1CustomResourceDefinitionNames.
        :type: str
        """
    self._singular = singular