from pprint import pformat
from six import iteritems
import re
@short_names.setter
def short_names(self, short_names):
    """
        Sets the short_names of this V1beta1CustomResourceDefinitionNames.
        ShortNames are short names for the resource.  It must be all lowercase.

        :param short_names: The short_names of this
        V1beta1CustomResourceDefinitionNames.
        :type: list[str]
        """
    self._short_names = short_names