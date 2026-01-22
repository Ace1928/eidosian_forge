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
def to_n_cpus(cpu_str: str) -> Union[int, float]:
    """Convert cpu string to number of cpus
    (e.g. '500m' -> 0.5, '2000000000n' -> 2)
    """
    if cpu_str.endswith('n'):
        return int(cpu_str.strip('n')) / 1000000000
    elif cpu_str.endswith('u'):
        return int(cpu_str.strip('u')) / 1000000
    elif cpu_str.endswith('m'):
        return int(cpu_str.strip('m')) / 1000
    elif cpu_str.isnumeric():
        return int(cpu_str)
    else:
        return 0