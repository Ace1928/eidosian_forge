import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def timeline(self, filename: Optional[str]=None) -> Optional[List[Any]]:
    logger.warning('Timeline will include events from other clients using this server.')
    import ray.core.generated.ray_client_pb2 as ray_client_pb2
    all_events = self.worker.get_cluster_info(ray_client_pb2.ClusterInfoType.TIMELINE)
    if filename is not None:
        with open(filename, 'w') as outfile:
            json.dump(all_events, outfile)
    else:
        return all_events