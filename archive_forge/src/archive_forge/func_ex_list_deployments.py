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
def ex_list_deployments(self) -> List[KubernetesDeployment]:
    """Get cluster deployments

        :rtype: ``list`` of :class:`.KubernetesDeployment`
        """
    items = self.connection.request('/apis/apps/v1/deployments').object['items']
    return [self._to_deployment(item) for item in items]