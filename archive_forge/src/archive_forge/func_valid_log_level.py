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
@validator('log_level')
def valid_log_level(cls, v):
    if isinstance(v, int):
        if v not in logging._levelToName:
            raise ValueError(f'Got "{v}" for log_level. log_level must be one of {list(logging._levelToName.keys())}.')
        return logging._levelToName[v]
    if v not in logging._nameToLevel:
        raise ValueError(f'Got "{v}" for log_level. log_level must be one of {list(logging._nameToLevel.keys())}.')
    return v