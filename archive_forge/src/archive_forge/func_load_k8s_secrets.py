import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def load_k8s_secrets() -> Tuple[Dict[str, str], str]:
    """
    Loads secrets needed to access K8s resources.

    Returns:
        headers: Headers with K8s access token
        verify: Path to certificate
    """
    with open('/var/run/secrets/kubernetes.io/serviceaccount/token') as secret:
        token = secret.read()
    headers = {'Authorization': 'Bearer ' + token}
    verify = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    return (headers, verify)