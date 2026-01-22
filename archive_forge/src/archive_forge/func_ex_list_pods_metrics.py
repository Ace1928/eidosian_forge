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
def ex_list_pods_metrics(self) -> List[Dict[str, Any]]:
    """Get pods metrics from Kubernetes Metrics Server

        :rtype: ``list`` of ``dict``
        """
    return self.connection.request('/apis/metrics.k8s.io/v1beta1/pods').object['items']