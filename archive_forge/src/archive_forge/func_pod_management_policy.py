from pprint import pformat
from six import iteritems
import re
@pod_management_policy.setter
def pod_management_policy(self, pod_management_policy):
    """
        Sets the pod_management_policy of this V1beta2StatefulSetSpec.
        podManagementPolicy controls how pods are created during initial scale
        up, when replacing pods on nodes, or when scaling down. The default
        policy is `OrderedReady`, where pods are created in increasing order
        (pod-0, then pod-1, etc) and the controller will wait until each pod is
        ready before continuing. When scaling down, the pods are removed in the
        opposite order. The alternative policy is `Parallel` which will create
        pods in parallel to match the desired scale without waiting, and on
        scale down will delete all pods at once.

        :param pod_management_policy: The pod_management_policy of this
        V1beta2StatefulSetSpec.
        :type: str
        """
    self._pod_management_policy = pod_management_policy