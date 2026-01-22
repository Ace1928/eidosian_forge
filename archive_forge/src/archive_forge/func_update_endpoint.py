import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def update_endpoint(self, endpoint: EndpointTag, endpoint_info: EndpointInfo) -> None:
    """Create or update the given endpoint.

        This method is idempotent - if the endpoint already exists it will be
        updated to match the given parameters. Calling this twice with the same
        arguments is a no-op.
        """
    if self._endpoints.get(endpoint) == endpoint_info:
        return
    existing_route_endpoint = self._get_endpoint_for_route(endpoint_info.route)
    if existing_route_endpoint is not None and existing_route_endpoint != endpoint:
        logger.debug(f'route_prefix "{endpoint_info.route}" is currently registered to deployment "{existing_route_endpoint.name}". Re-registering route_prefix "{endpoint_info.route}" to deployment "{endpoint.name}".')
        del self._endpoints[existing_route_endpoint]
    self._endpoints[endpoint] = endpoint_info
    self._checkpoint()
    self._notify_route_table_changed()