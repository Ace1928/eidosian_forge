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
@validator('import_path')
def import_path_format_valid(cls, v: str):
    if v is None:
        return
    if ':' in v:
        if v.count(':') > 1:
            raise ValueError(f'Got invalid import path "{v}". An import path may have at most one colon.')
        if v.rfind(':') == 0 or v.rfind(':') == len(v) - 1:
            raise ValueError(f'Got invalid import path "{v}". An import path may not start or end with a colon.')
        return v
    else:
        if v.count('.') < 1:
            raise ValueError(f'Got invalid import path "{v}". An import path must contain at least on dot or colon separating the module (and potentially submodules) from the deployment graph. E.g.: "module.deployment_graph".')
        if v.rfind('.') == 0 or v.rfind('.') == len(v) - 1:
            raise ValueError(f'Got invalid import path "{v}". An import path may not start or end with a dot.')
    return v