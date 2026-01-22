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
def job_table(self):
    """Fetch and parse the gcs job table.

        Returns:
            Information about the Ray jobs in the cluster,
            namely a list of dicts with keys:
            - "JobID" (identifier for the job),
            - "DriverIPAddress" (IP address of the driver for this job),
            - "DriverPid" (process ID of the driver for this job),
            - "StartTime" (UNIX timestamp of the start time of this job),
            - "StopTime" (UNIX timestamp of the stop time of this job, if any)
        """
    self._check_connected()
    job_table = self.global_state_accessor.get_job_table()
    results = []
    for i in range(len(job_table)):
        entry = gcs_pb2.JobTableData.FromString(job_table[i])
        job_info = {}
        job_info['JobID'] = entry.job_id.hex()
        job_info['DriverIPAddress'] = entry.driver_address.ip_address
        job_info['DriverPid'] = entry.driver_pid
        job_info['Timestamp'] = entry.timestamp
        job_info['StartTime'] = entry.start_time
        job_info['EndTime'] = entry.end_time
        job_info['IsDead'] = entry.is_dead
        job_info['Entrypoint'] = entry.entrypoint
        results.append(job_info)
    return results