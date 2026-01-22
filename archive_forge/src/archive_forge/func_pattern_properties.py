from pprint import pformat
from six import iteritems
import re
@pattern_properties.setter
def pattern_properties(self, pattern_properties):
    """
        Sets the pattern_properties of this V1beta1JSONSchemaProps.

        :param pattern_properties: The pattern_properties of this
        V1beta1JSONSchemaProps.
        :type: dict(str, V1beta1JSONSchemaProps)
        """
    self._pattern_properties = pattern_properties