from pprint import pformat
from six import iteritems
import re
@required_drop_capabilities.setter
def required_drop_capabilities(self, required_drop_capabilities):
    """
        Sets the required_drop_capabilities of this
        PolicyV1beta1PodSecurityPolicySpec.
        requiredDropCapabilities are the capabilities that will be dropped from
        the container.  These are required to be dropped and cannot be added.

        :param required_drop_capabilities: The required_drop_capabilities of
        this PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._required_drop_capabilities = required_drop_capabilities