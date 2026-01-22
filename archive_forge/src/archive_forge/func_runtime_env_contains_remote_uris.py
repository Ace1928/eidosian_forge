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
@validator('runtime_env')
def runtime_env_contains_remote_uris(cls, v):
    if v is None:
        return
    uris = v.get('py_modules', [])
    if 'working_dir' in v and v['working_dir'] not in uris:
        uris.append(v['working_dir'])
    for uri in uris:
        if uri is not None:
            try:
                parse_uri(uri)
            except ValueError as e:
                raise ValueError(f'runtime_envs in the Serve config support only remote URIs in working_dir and py_modules. Got error when parsing URI: {e}')
    return v