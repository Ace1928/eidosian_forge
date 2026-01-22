import time
from collections import defaultdict
from typing import List
from ray._raylet import GcsClient
from ray.autoscaler.v2.schema import ClusterStatus, Stats
from ray.autoscaler.v2.utils import ClusterStatusParser
from ray.core.generated.autoscaler_pb2 import GetClusterStatusReply
def request_cluster_resources(gcs_address: str, to_request: List[dict], timeout: int=DEFAULT_RPC_TIMEOUT_S):
    """Request resources from the autoscaler.

    This will add a cluster resource constraint to GCS. GCS will asynchronously
    pass the constraint to the autoscaler, and the autoscaler will try to provision the
    requested minimal bundles in `to_request`.

    If the cluster already has `to_request` resources, this will be an no-op.
    Future requests submitted through this API will overwrite the previous requests.

    Args:
        gcs_address: The GCS address to query.
        to_request: A list of resource bundles to request the cluster to have.
            Each bundle is a dict of resource name to resource quantity, e.g:
            [{"CPU": 1}, {"GPU": 1}].
        timeout: Timeout in seconds for the request to be timeout

    """
    assert len(gcs_address) > 0, 'GCS address is not specified.'
    resource_requests_by_count = defaultdict(int)
    for request in to_request:
        bundle = frozenset(request.items())
        resource_requests_by_count[bundle] += 1
    bundles = []
    counts = []
    for bundle, count in resource_requests_by_count.items():
        bundles.append(dict(bundle))
        counts.append(count)
    GcsClient(gcs_address).request_cluster_resource_constraint(bundles, counts, timeout_s=timeout)