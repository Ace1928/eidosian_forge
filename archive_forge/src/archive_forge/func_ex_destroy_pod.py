import json
import hashlib
import datetime
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.types import NodeState
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.kubernetes import (
from libcloud.container.providers import Provider
def ex_destroy_pod(self, namespace: str, pod_name: str) -> bool:
    """
        Delete a pod and the containers within it.

        :param namespace: The pod's namespace
        :type  namespace: ``str``

        :param pod_name: Name of the pod to destroy
        :type  pod_name: ``str``

        :rtype: ``bool``
        """
    self.connection.request(ROOT_URL + 'v1/namespaces/{}/pods/{}'.format(namespace, pod_name), method='DELETE').object
    return True