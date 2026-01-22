import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@root_validator
def num_replicas_and_autoscaling_config_mutually_exclusive(cls, values):
    if values.get('num_replicas', None) not in [DEFAULT.VALUE, None] and values.get('autoscaling_config', None) not in [DEFAULT.VALUE, None]:
        raise ValueError('Manually setting num_replicas is not allowed when autoscaling_config is provided.')
    return values