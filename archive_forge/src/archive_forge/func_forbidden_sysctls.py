from pprint import pformat
from six import iteritems
import re
@forbidden_sysctls.setter
def forbidden_sysctls(self, forbidden_sysctls):
    """
        Sets the forbidden_sysctls of this PolicyV1beta1PodSecurityPolicySpec.
        forbiddenSysctls is a list of explicitly forbidden sysctls, defaults to
        none. Each entry is either a plain sysctl name or ends in "*" in which
        case it is considered as a prefix of forbidden sysctls. Single * means
        all sysctls are forbidden.  Examples: e.g. "foo/*" forbids
        "foo/bar", "foo/baz", etc. e.g. "foo.*" forbids "foo.bar",
        "foo.baz", etc.

        :param forbidden_sysctls: The forbidden_sysctls of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._forbidden_sysctls = forbidden_sysctls