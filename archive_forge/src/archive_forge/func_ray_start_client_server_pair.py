from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook
@contextmanager
def ray_start_client_server_pair(metadata=None, ray_connect_handler=None, **kwargs):
    ray._inside_client_test = True
    with disable_client_hook():
        assert not ray.is_initialized()
    server = ray_client_server.serve('127.0.0.1:50051', ray_connect_handler=ray_connect_handler)
    ray.connect('127.0.0.1:50051', metadata=metadata, **kwargs)
    try:
        yield (ray, server)
    finally:
        ray._inside_client_test = False
        ray.disconnect()
        server.stop(0)
        del server
        start = time.monotonic()
        with disable_client_hook():
            while ray.is_initialized():
                time.sleep(1)
                if time.monotonic() - start > 30:
                    raise RuntimeError('Failed to terminate Ray')
        time.sleep(3)