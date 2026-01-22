import ray
import ray._private.profiling as profiling
import ray._private.services as services
import ray._private.utils as utils
import ray._private.worker
from ray._private import ray_constants
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
def global_gc():
    """Trigger gc.collect() on all workers in the cluster."""
    worker = ray._private.worker.global_worker
    worker.core_worker.global_gc()