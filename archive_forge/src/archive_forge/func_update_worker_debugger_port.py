import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def update_worker_debugger_port(self, worker_id, debugger_port):
    """Update the debugger port of a worker.

        Args:
            worker_id: ID of this worker. Type is bytes.
            debugger_port: Port of the debugger. Type is int.

        Returns:
             Is operation success
        """
    self._check_connected()
    assert worker_id is not None, 'worker_id is not valid'
    assert debugger_port is not None and debugger_port > 0, 'debugger_port is not valid'
    return self.global_state_accessor.update_worker_debugger_port(worker_id, debugger_port)