import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def get_endpoint_route(self, endpoint: EndpointTag) -> Optional[str]:
    if endpoint in self._endpoints:
        return self._endpoints[endpoint].route
    return None