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
def nested_host_and_port(cls, values):
    for app_config in values.get('applications'):
        if 'host' in app_config.dict(exclude_unset=True):
            raise ValueError(f'Host "{app_config.host}" is set in the config for application `{app_config.name}`. Please remove it and set host in the top level deploy config only.')
        if 'port' in app_config.dict(exclude_unset=True):
            raise ValueError(f'Port {app_config.port} is set in the config for application `{app_config.name}`. Please remove it and set port in the top level deploy config only.')
    return values