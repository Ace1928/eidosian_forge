from pprint import pformat
from six import iteritems
import re
@policy_types.setter
def policy_types(self, policy_types):
    """
        Sets the policy_types of this V1NetworkPolicySpec.
        List of rule types that the NetworkPolicy relates to. Valid options are
        "Ingress", "Egress", or "Ingress,Egress". If this field is not
        specified, it will default based on the existence of Ingress or Egress
        rules; policies that contain an Egress section are assumed to affect
        Egress, and all policies (whether or not they contain an Ingress
        section) are assumed to affect Ingress. If you want to write an
        egress-only policy, you must explicitly specify policyTypes [ "Egress"
        ]. Likewise, if you want to write a policy that specifies that no egress
        is allowed, you must specify a policyTypes value that include "Egress"
        (since such a policy would not include an Egress section and would
        otherwise default to just [ "Ingress" ]). This field is beta-level in
        1.8

        :param policy_types: The policy_types of this V1NetworkPolicySpec.
        :type: list[str]
        """
    self._policy_types = policy_types