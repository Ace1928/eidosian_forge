import asyncio
import concurrent.futures
import threading
import warnings
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union
import ray
from ray import serve
from ray._raylet import GcsClient, ObjectRefGenerator
from ray.serve._private.common import DeploymentID, RequestProtocol
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.router import RequestMetadata, Router
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.util import metrics
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def copy_and_update(self, method_name: Union[str, DEFAULT]=DEFAULT.VALUE, multiplexed_model_id: Union[str, DEFAULT]=DEFAULT.VALUE, stream: Union[bool, DEFAULT]=DEFAULT.VALUE, _prefer_local_routing: Union[bool, DEFAULT]=DEFAULT.VALUE, _router_cls: Union[str, DEFAULT]=DEFAULT.VALUE, _request_protocol: Union[str, DEFAULT]=DEFAULT.VALUE) -> '_HandleOptions':
    return _HandleOptions(method_name=self.method_name if method_name == DEFAULT.VALUE else method_name, multiplexed_model_id=self.multiplexed_model_id if multiplexed_model_id == DEFAULT.VALUE else multiplexed_model_id, stream=self.stream if stream == DEFAULT.VALUE else stream, _prefer_local_routing=self._prefer_local_routing if _prefer_local_routing == DEFAULT.VALUE else _prefer_local_routing, _router_cls=self._router_cls if _router_cls == DEFAULT.VALUE else _router_cls, _request_protocol=self._request_protocol if _request_protocol == DEFAULT.VALUE else _request_protocol)