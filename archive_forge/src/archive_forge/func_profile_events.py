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
def profile_events(self):
    """Retrieve and return task profiling events from GCS.

        Return:
            Profiling events by component id (e.g. worker id).
            {
                <component_id>: [
                    {
                        event_type: <event name> ,
                        component_id: <i.e. worker id>,
                        node_ip_address: <on which node profiling was done>,
                        component_type: <i.e. worker/driver>,
                        start_time: <unix timestamp in seconds>,
                        end_time: <unix timestamp in seconds>,
                        extra_data: <e.g. stack trace when error raised>,
                    }
                ]
            }
        """
    self._check_connected()
    result = defaultdict(list)
    task_events = self.global_state_accessor.get_task_events()
    for i in range(len(task_events)):
        event = gcs_pb2.TaskEvents.FromString(task_events[i])
        profile = event.profile_events
        if not profile:
            continue
        component_type = profile.component_type
        component_id = binary_to_hex(profile.component_id)
        node_ip_address = profile.node_ip_address
        for event in profile.events:
            try:
                extra_data = json.loads(event.extra_data)
            except ValueError:
                extra_data = {}
            profile_event = {'event_type': event.event_name, 'component_id': component_id, 'node_ip_address': node_ip_address, 'component_type': component_type, 'start_time': event.start_time, 'end_time': event.end_time, 'extra_data': extra_data}
            result[component_id].append(profile_event)
    return dict(result)