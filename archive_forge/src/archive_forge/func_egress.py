from pprint import pformat
from six import iteritems
import re
@egress.setter
def egress(self, egress):
    """
        Sets the egress of this V1NetworkPolicySpec.
        List of egress rules to be applied to the selected pods. Outgoing
        traffic is allowed if there are no NetworkPolicies selecting the pod
        (and cluster policy otherwise allows the traffic), OR if the traffic
        matches at least one egress rule across all of the NetworkPolicy objects
        whose podSelector matches the pod. If this field is empty then this
        NetworkPolicy limits all outgoing traffic (and serves solely to ensure
        that the pods it selects are isolated by default). This field is
        beta-level in 1.8

        :param egress: The egress of this V1NetworkPolicySpec.
        :type: list[V1NetworkPolicyEgressRule]
        """
    self._egress = egress