import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]):
    logger.info(f'Got updated endpoints: {endpoints}.', extra={'log_to_stderr': False})
    self.endpoints = endpoints
    existing_handles = set(self.handles.keys())
    for endpoint, info in endpoints.items():
        if endpoint in self.handles:
            existing_handles.remove(endpoint)
        else:
            handle = self._get_handle(endpoint.name, endpoint.app).options(stream=not info.app_is_cross_language, use_new_handle_api=True, _prefer_local_routing=RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
            handle._set_request_protocol(self._protocol)
            self.handles[endpoint] = handle
    if len(existing_handles) > 0:
        logger.info(f'Deleting {len(existing_handles)} unused handles.', extra={'log_to_stderr': False})
    for endpoint in existing_handles:
        del self.handles[endpoint]