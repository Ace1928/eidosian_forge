import json
import logging
from pathlib import Path
from threading import RLock
from uuid import uuid4
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
from ray.autoscaler._private._azure.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def match_tags(vm):
    for k, v in cluster_tag_filters.items():
        if vm.tags.get(k) != v:
            return False
    return True