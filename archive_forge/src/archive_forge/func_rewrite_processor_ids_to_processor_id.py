import datetime
import sys
from typing import (
import warnings
import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager
def rewrite_processor_ids_to_processor_id(args, kwargs):
    """Rewrites the create_job parameters so that `processor_id` is used instead of the deprecated
    `processor_ids`.

        Raises:
            ValueError: If `processor_ids` has more than one processor id.
            ValueError: If `run_name` or `device_config_name` are set but `processor_id` is not.
    """
    processor_ids = args[4] if len(args) > 4 else kwargs['processor_ids']
    if len(processor_ids) > 1:
        raise ValueError('The use of multiple processors is no longer supported.')
    if 'processor_id' not in kwargs or not kwargs['processor_id']:
        if 'run_name' in kwargs and kwargs['run_name'] or ('device_config_name' in kwargs and kwargs['device_config_name']):
            raise ValueError('Cannot specify `run_name` or `device_config_name` if `processor_id` is empty.')
        kwargs['processor_id'] = processor_ids[0]
    if len(args) > 4:
        args_list = list(args)
        args_list[4] = None
        args = tuple(args_list)
    else:
        kwargs.pop('processor_ids')
    return (args, kwargs)