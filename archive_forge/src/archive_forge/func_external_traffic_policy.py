from pprint import pformat
from six import iteritems
import re
@external_traffic_policy.setter
def external_traffic_policy(self, external_traffic_policy):
    """
        Sets the external_traffic_policy of this V1ServiceSpec.
        externalTrafficPolicy denotes if this Service desires to route external
        traffic to node-local or cluster-wide endpoints. "Local" preserves the
        client source IP and avoids a second hop for LoadBalancer and Nodeport
        type services, but risks potentially imbalanced traffic spreading.
        "Cluster" obscures the client source IP and may cause a second hop to
        another node, but should have good overall load-spreading.

        :param external_traffic_policy: The external_traffic_policy of this
        V1ServiceSpec.
        :type: str
        """
    self._external_traffic_policy = external_traffic_policy