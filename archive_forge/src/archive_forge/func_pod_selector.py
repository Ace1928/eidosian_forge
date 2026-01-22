from pprint import pformat
from six import iteritems
import re
@pod_selector.setter
def pod_selector(self, pod_selector):
    """
        Sets the pod_selector of this V1beta1NetworkPolicyPeer.
        This is a label selector which selects Pods. This field follows standard
        label selector semantics; if present but empty, it selects all pods.  If
        NamespaceSelector is also set, then the NetworkPolicyPeer as a whole
        selects the Pods matching PodSelector in the Namespaces selected by
        NamespaceSelector. Otherwise it selects the Pods matching PodSelector in
        the policy's own Namespace.

        :param pod_selector: The pod_selector of this V1beta1NetworkPolicyPeer.
        :type: V1LabelSelector
        """
    self._pod_selector = pod_selector