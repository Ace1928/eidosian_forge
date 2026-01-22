from pprint import pformat
from six import iteritems
import re
@namespace_selector.setter
def namespace_selector(self, namespace_selector):
    """
        Sets the namespace_selector of this V1beta1NetworkPolicyPeer.
        Selects Namespaces using cluster-scoped labels. This field follows
        standard label selector semantics; if present but empty, it selects all
        namespaces.  If PodSelector is also set, then the NetworkPolicyPeer as a
        whole selects the Pods matching PodSelector in the Namespaces selected
        by NamespaceSelector. Otherwise it selects all Pods in the Namespaces
        selected by NamespaceSelector.

        :param namespace_selector: The namespace_selector of this
        V1beta1NetworkPolicyPeer.
        :type: V1LabelSelector
        """
    self._namespace_selector = namespace_selector