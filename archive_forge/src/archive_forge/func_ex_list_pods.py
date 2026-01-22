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
def ex_list_pods(self, fetch_metrics: bool=False) -> List[KubernetesPod]:
    """
        List available Pods

        :param fetch_metrics: Fetch metrics for pods
        :type  fetch_metrics: ``bool``

        :rtype: ``list`` of :class:`.KubernetesPod`
        """
    result = self.connection.request(ROOT_URL + 'v1/pods').object
    metrics = None
    if fetch_metrics:
        try:
            metrics = {(metric['metadata']['name'], metric['metadata']['namespace']): metric['containers'] for metric in self.ex_list_pods_metrics()}
        except BaseHTTPError:
            pass
    return [self._to_pod(value, metrics=metrics) for value in result['items']]