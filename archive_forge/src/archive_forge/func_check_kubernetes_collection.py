from __future__ import (absolute_import, division, print_function)
def check_kubernetes_collection(self):
    if not HAS_KUBERNETES_COLLECTION:
        K8sInventoryException('The kubernetes.core collection must be installed')