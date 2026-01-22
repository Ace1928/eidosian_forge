import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def call_remote(self, instance: 'ClientStub', *args, **kwargs) -> List[Future]:
    """call_remote is called by stub objects to execute them remotely.

        This is used by stub objects in situations where they're called
        with .remote, eg, `f.remote()` or `actor_cls.remote()`.
        This allows the client stub objects to delegate execution to be
        implemented in the most effective way whether it's in the client,
        clientserver, or raylet worker.

        Args:
            instance: The Client-side stub reference to a remote object
            args: opaque arguments
            kwargs: opaque keyword arguments
        """
    return self.worker.call_remote(instance, *args, **kwargs)