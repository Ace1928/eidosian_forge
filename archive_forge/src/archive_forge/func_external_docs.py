from pprint import pformat
from six import iteritems
import re
@external_docs.setter
def external_docs(self, external_docs):
    """
        Sets the external_docs of this V1beta1JSONSchemaProps.

        :param external_docs: The external_docs of this V1beta1JSONSchemaProps.
        :type: V1beta1ExternalDocumentation
        """
    self._external_docs = external_docs