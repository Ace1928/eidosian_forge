from pprint import pformat
from six import iteritems
import re
@se_linux.setter
def se_linux(self, se_linux):
    """
        Sets the se_linux of this PolicyV1beta1PodSecurityPolicySpec.
        seLinux is the strategy that will dictate the allowable labels that may
        be set.

        :param se_linux: The se_linux of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: PolicyV1beta1SELinuxStrategyOptions
        """
    if se_linux is None:
        raise ValueError('Invalid value for `se_linux`, must not be `None`')
    self._se_linux = se_linux