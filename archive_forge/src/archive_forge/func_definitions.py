from pprint import pformat
from six import iteritems
import re
@definitions.setter
def definitions(self, definitions):
    """
        Sets the definitions of this V1beta1JSONSchemaProps.

        :param definitions: The definitions of this V1beta1JSONSchemaProps.
        :type: dict(str, V1beta1JSONSchemaProps)
        """
    self._definitions = definitions